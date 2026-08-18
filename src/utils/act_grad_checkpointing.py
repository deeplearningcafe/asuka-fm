import torch
import os
from torch.utils.checkpoint import (
    _infer_device_type,
    _get_autocast_kwargs,
    _get_device_module,
    get_device_states,
    contextlib,
    DefaultDeviceType,
    ContextManager,
    _DEFAULT_DETERMINISM_MODE,
    _checkpoint_without_reentrant_generator,
    noop_context_fn,
)
from typing import Optional, Callable, Tuple
import warnings


class CPUGradientAccumulator:
    """
    High-performance CPU Gradient Accumulator using CUDA streams and
    pinned memory buffers. Eliminates background worker threads and queues
    to prevent GPU starvation under high host CPU utilization.
    """

    def __init__(self, model):
        self.model = model
        self.grad_buffers = {}
        self.staging_buffers = {}

        self.copy_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        self.trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.num_params = len(self.trainable_params)
        self.hook_counter = 0
        self.micro_step = 0

        for param in self.trainable_params:
            # Pre-allocate pinned CPU buffers for zero-copy DMA transfers
            self.grad_buffers[param] = torch.zeros(
                param.shape,
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            self.staging_buffers[param] = torch.zeros(
                param.shape,
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )

            param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _make_hook(self, param):
        def hook_fn(p):
            if p.grad is not None:
                if self.copy_stream:
                    # Align copy stream with compute stream execution
                    self.copy_stream.wait_stream(torch.cuda.current_stream())

                    with torch.cuda.stream(self.copy_stream):
                        if self.micro_step == 0:
                            # Direct DMA copy to main CPU buffer on step 0
                            self.grad_buffers[p].copy_(p.grad, non_blocking=True)
                        else:
                            # DMA copy to staging buffer on microstep > 0
                            self.staging_buffers[p].copy_(p.grad, non_blocking=True)

                    # Tell allocator memory is in use by copy_stream
                    p.grad.record_stream(self.copy_stream)

                    if self.micro_step > 0:
                        self.copy_stream.synchronize()
                        self.grad_buffers[p].add_(self.staging_buffers[p])
                else:
                    if self.micro_step == 0:
                        self.grad_buffers[p].copy_(p.grad)
                    else:
                        self.grad_buffers[p].add_(p.grad.to("cpu"))

                # Free GPU gradient tensor memory immediately
                p.grad = None

                self.hook_counter += 1
                if self.hook_counter == self.num_params:
                    self.hook_counter = 0
                    self.micro_step += 1

        return hook_fn

    def finalize_and_step(self, optimizer, scaler=None, max_norm=1.0):
        if self.copy_stream:
            # Wait for any remaining microstep DMA transfers to finish
            self.copy_stream.synchronize()

            # Copy CPU accumulated gradients to GPU asynchronously
            with torch.cuda.stream(self.copy_stream):
                for param, cpu_grad in self.grad_buffers.items():
                    param.grad = torch.empty_like(param)
                    param.grad.copy_(cpu_grad, non_blocking=True)

            # Synchronize CUDA streams on GPU side (0 CPU waiting)
            torch.cuda.current_stream().wait_stream(self.copy_stream)
        else:
            for param, cpu_grad in self.grad_buffers.items():
                param.grad = cpu_grad.to(param.device)

        self.hook_counter = 0
        self.micro_step = 0

        # Synchronize gradients across DDP ranks if active
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            for param in self.grad_buffers.keys():
                if param.grad is not None:
                    torch.distributed.all_reduce(
                        param.grad, op=torch.distributed.ReduceOp.SUM
                    )
                    param.grad.div_(world_size)

        if scaler is not None:
            scaler.unscale_(optimizer)

        total_norm = torch.nn.utils.clip_grad_norm_(self.grad_buffers.keys(), max_norm)

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        return total_norm


# copied from https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/gradient_checkpointing.py
# Added [device_type] in Torch 2.5!
def set_device_states(devices, states, *, device_type=None) -> None:
    """Sets random number generator states for the specified devices.

    Args:
        devices: Device ids to set states for.
        states: States to set.
        device_type: ``device_type`` of the devices to set states for. Default
            is the device returned by a call to ``DefaultDeviceType.get_device_type()``,
            which is ``cuda`` if not changed by calling ``DefaultDeviceType::set_device_type()``.
    """
    if device_type is None:
        device_type = DefaultDeviceType.get_device_type()
    if device_type == "meta":
        return
    device_module = _get_device_module(device_type)
    for device, state in zip(devices, states):
        with device_module.device(device):
            device_module.set_rng_state(state)


def get_device_type():
    import torch

    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    raise NotImplementedError(
        "Unsloth currently only works on NVIDIA GPUs and Intel GPUs."
    )


pass
DEVICE_TYPE: str = get_device_type()


global CPU_BUFFERS
global CPU_INDEX
global GPU_BUFFERS
global BACKWARD_PASS
global EXTRA_STREAMS
global MAIN_STREAMS
global MINIMUM_SIZE
global USE_UNSLOTH_GC
global LAST_GC_INDEX
global FIRST_PASS
global CURRENT_GC_INDEX

if DEVICE_TYPE == "cuda":
    torch_gpu_stream = torch.cuda.stream
elif DEVICE_TYPE == "xpu":
    torch_gpu_stream = torch.xpu.stream

CPU_BUFFERS = []
CPU_INDEX = None


def initialize_unsloth_gradient_checkpointing(dtype=None):
    # All Unsloth Zoo code licensed under LGPLv3
    global CPU_BUFFERS
    global CPU_INDEX
    global GPU_BUFFERS
    global BACKWARD_PASS
    global EXTRA_STREAMS
    global MAIN_STREAMS
    global MINIMUM_SIZE
    global USE_UNSLOTH_GC
    global LAST_GC_INDEX
    global FIRST_PASS
    global CURRENT_GC_INDEX
    CPU_BUFFERS = []
    CPU_INDEX = 0

    if dtype is None:
        if DEVICE_TYPE == "cuda":
            major_version, minor_version = torch.cuda.get_device_capability()
            SUPPORTS_BFLOAT16 = major_version >= 8
        elif DEVICE_TYPE == "xpu":
            SUPPORTS_BFLOAT16 = True
        dtype = torch.bfloat16 if SUPPORTS_BFLOAT16 else torch.float16
    pass

    for i in range(200):
        x = torch.empty(128 * 1024, dtype=dtype, device="cpu", pin_memory=True)
        CPU_BUFFERS.append(x)
    pass

    # Determine total visible GPUs in the environment
    n_gpus = (
        torch.cuda.device_count() if DEVICE_TYPE == "cuda" else torch.xpu.device_count()
    )

    # Fetch the active device index set for the current process
    if DEVICE_TYPE == "cuda":
        current_device_idx = torch.cuda.current_device()
    elif DEVICE_TYPE == "xpu":
        current_device_idx = torch.xpu.current_device()
    else:
        current_device_idx = 0

    # Initialize a sparse container to allocate buffers only on the active GPU
    GPU_BUFFERS = [None] * n_gpus
    GPU_BUFFERS[current_device_idx] = torch.empty(
        2 * 256 * 2048, dtype=dtype, device=f"{DEVICE_TYPE}:{current_device_idx}"
    )
    GPU_BUFFERS = tuple(GPU_BUFFERS)

    BACKWARD_PASS = True

    # Allocate a stream only on the process's active GPU
    EXTRA_STREAMS = [None] * n_gpus
    if DEVICE_TYPE == "cuda":
        EXTRA_STREAMS[current_device_idx] = torch.cuda.Stream(device=current_device_idx)
    elif DEVICE_TYPE == "xpu":
        EXTRA_STREAMS[current_device_idx] = torch.xpu.Stream(device=current_device_idx)
    EXTRA_STREAMS = tuple(EXTRA_STREAMS)

    # Bind default streams exclusively on the process's active GPU
    MAIN_STREAMS = [None] * n_gpus
    if DEVICE_TYPE == "cuda":
        MAIN_STREAMS[current_device_idx] = torch.cuda.default_stream(
            torch.device(f"cuda:{current_device_idx}")
        )
    elif DEVICE_TYPE == "xpu":
        MAIN_STREAMS[current_device_idx] = torch.xpu.current_stream(
            torch.device(f"xpu:{current_device_idx}")
        )
    MAIN_STREAMS = tuple(MAIN_STREAMS)

    # Minimum size to enable Unsloth GC is 2MB -> 32 layers = 64MB
    n_bytes = torch.finfo(dtype).bits // 8
    MINIMUM_SIZE = 2 * 1024 * 1024 // n_bytes
    USE_UNSLOTH_GC = True

    # Disable offloading on the last layer - uses more VRAM and is slower
    # See https://github.com/pytorch/torchtune/pull/1443
    LAST_GC_INDEX = 0
    FIRST_PASS = True
    CURRENT_GC_INDEX = 0


class UnslothCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        # All Unsloth Zoo code licensed under LGPLv3
        # check_backward_validity(args)
        # Check if no requires_grad in inputs
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.device_type = _infer_device_type(*args)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device_type
        )
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device_type)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        ctx._requires_gradient = False
        use_gpu_buffer = False

        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                if i == 0 and arg.requires_grad:
                    global FIRST_PASS
                    global LAST_GC_INDEX
                    if FIRST_PASS:
                        # Save last layer index so next run we do not offload activations
                        # Saves VRAM and saves some time
                        # See https://github.com/pytorch/torchtune/pull/1443
                        LAST_GC_INDEX += 1
                    pass
                    global CURRENT_GC_INDEX
                    CURRENT_GC_INDEX += 1

                    ctx._requires_gradient = True
                    new_size = arg.numel()

                    global MINIMUM_SIZE
                    global CPU_INDEX
                    if new_size > MINIMUM_SIZE and CURRENT_GC_INDEX != LAST_GC_INDEX:
                        use_gpu_buffer = True
                        global CPU_BUFFERS
                        global GPU_BUFFERS
                        global BACKWARD_PASS
                        global EXTRA_STREAMS
                        global MAIN_STREAMS
                        device = arg.device
                        device_index = device.index
                        GPU_BUFFER = GPU_BUFFERS[device_index]
                        MAIN_STREAM = MAIN_STREAMS[device_index]
                        EXTRA_STREAM = EXTRA_STREAMS[device_index]

                        # Handle interrupted training runs
                        if BACKWARD_PASS:
                            BACKWARD_PASS = False
                            CPU_INDEX = 0
                        pass

                        # Extend buffer size
                        if CPU_INDEX >= len(CPU_BUFFERS):
                            x = torch.empty(
                                new_size, dtype=arg.dtype, device="cpu", pin_memory=True
                            )
                            CPU_BUFFERS.append(x)
                        pass

                        x = CPU_BUFFERS[CPU_INDEX]
                        shape = arg.shape
                        if new_size > x.numel():
                            x.resize_(new_size)
                        if new_size > GPU_BUFFER.numel():
                            GPU_BUFFER.resize_(new_size)
                        x = x[:new_size].view(shape)

                        # See https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
                        EXTRA_STREAM.wait_stream(MAIN_STREAM)
                        with torch_gpu_stream(EXTRA_STREAM):
                            x.copy_(arg, non_blocking=True)

                        # CRITICAL: Ensure `arg` is not freed by the caching allocator
                        # before the async copy on EXTRA_STREAM completes.
                        arg.record_stream(EXTRA_STREAM)

                        ctx._saved_metadata = (
                            new_size,
                            shape,
                            CPU_INDEX,
                            device_index,
                            MAIN_STREAM,
                            EXTRA_STREAM,
                        )
                        CPU_INDEX += 1
                        tensor_inputs.append(None)

                        global USE_UNSLOTH_GC
                        if USE_UNSLOTH_GC:
                            print(
                                "Unsloth: Will smartly offload gradients to save VRAM!"
                            )
                            USE_UNSLOTH_GC = False
                    else:
                        ctx._saved_metadata = (
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        )
                        tensor_inputs.append(arg)
                    pass
                else:
                    tensor_inputs.append(arg)
                pass
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)
            pass
        pass
        if ctx._requires_gradient:
            ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)

        if use_gpu_buffer:
            MAIN_STREAM.wait_stream(EXTRA_STREAM)
        return outputs

    pass

    @staticmethod
    def backward(ctx, *args):
        # All Unsloth Zoo code licensed under LGPLv3
        if not ctx._requires_gradient:
            return None

        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "When use_reentrant=True, torch.utils.checkpoint is incompatible"
                " with .grad() or passing an `inputs` parameter to .backward()."
                " To resolve this error, you can either set use_reentrant=False,"
                " or call .backward() without passing the `inputs` argument."
            )

        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        new_size, shape, CPU_INDEX, device_index, MAIN_STREAM, EXTRA_STREAM = (
            ctx._saved_metadata
        )
        if CPU_INDEX is not None:
            global GPU_BUFFER
            buffer = GPU_BUFFERS[device_index][:new_size].view(shape)
            x = CPU_BUFFERS[CPU_INDEX][:new_size].view(shape)

            # See https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
            EXTRA_STREAM.wait_stream(MAIN_STREAM)
            with torch_gpu_stream(EXTRA_STREAM):
                buffer.copy_(x, non_blocking=True)
        else:
            # No GPU buffer seen
            if len(tensor_indices) != 0:
                inputs[tensor_indices[0]] = tensors[0]
        pass

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices[1:], start=1):
            inputs[idx] = tensors[i]
        pass

        global BACKWARD_PASS
        BACKWARD_PASS = True
        global FIRST_PASS
        FIRST_PASS = False
        global CURRENT_GC_INDEX
        CURRENT_GC_INDEX = 0

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices,
            enabled=ctx.preserve_rng_state,
            device_type=ctx.device_type,
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(
                        ctx.fwd_devices,
                        ctx.fwd_device_states,
                        device_type=ctx.device_type,
                    )

            device_autocast_ctx = (
                torch.amp.autocast(
                    device_type=ctx.device_type, **ctx.device_autocast_kwargs
                )
                if torch.amp.is_autocast_available(ctx.device_type)
                else contextlib.nullcontext()
            )

            # detached_inputs = detach_variable(tuple(inputs))
            detached_inputs = []
            for inp in inputs:
                if not isinstance(inp, torch.Tensor):
                    detached_inputs.append(inp)
                    continue
                x = inp.detach()
                x.requires_grad = inp.requires_grad
                detached_inputs.append(x)
            pass

            # Wait for GPU buffer to finish
            if CPU_INDEX is not None:
                MAIN_STREAM.wait_stream(EXTRA_STREAM)
                x = buffer.detach()
                x.requires_grad_(True)
                detached_inputs[0] = x
            pass

            with (
                torch.enable_grad(),
                device_autocast_ctx,
                torch.amp.autocast("cpu", **ctx.cpu_autocast_kwargs),
            ):  # type: ignore[attr-defined]
                outputs = ctx.run_function(*detached_inputs)
            pass
        pass

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        pass

        if len(outputs_with_grad) == 0:
            pass
            # raise RuntimeError(
            #     "none of output has requires_grad=True,"
            #     " this checkpoint() is not necessary"
            # )
        else:
            torch.autograd.backward(outputs_with_grad, args_with_grad)
        pass

        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in detached_inputs
        )
        # Clear all memory
        for i in range(len(detached_inputs)):
            detached_inputs[i] = None
            inputs[i] = None
        pass

        return (None, None) + grads

    pass


pass


@torch._disable_dynamo
def unsloth_checkpoint(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    **kwargs,
):
    r"""Checkpoint a model or part of the model.

    Activation checkpointing is a technique that trades compute for memory.
    Instead of keeping tensors needed for backward alive until they are used in
    gradient computation during backward, forward computation in checkpointed
    regions omits saving tensors for backward and recomputes them during the
    backward pass. Activation checkpointing can be applied to any part of a
    model.

    There are currently two checkpointing implementations available, determined
    by the :attr:`use_reentrant` parameter. It is recommended that you use
    ``use_reentrant=False``. Please refer the note below for a discussion of
    their differences.

    .. warning::

        If the :attr:`function` invocation during the backward pass differs
        from the forward pass, e.g., due to a global variable, the checkpointed
        version may not be equivalent, potentially causing an
        error being raised or leading to silently incorrect gradients.

    .. warning::

        The ``use_reentrant`` parameter should be passed explicitly. In version
        2.4 we will raise an exception if ``use_reentrant`` is not passed.
        If you are using the ``use_reentrant=True`` variant, please refer to the
        note below for important considerations and potential limitations.

    .. note::

        The reentrant variant of checkpoint (``use_reentrant=True``) and
        the non-reentrant variant of checkpoint (``use_reentrant=False``)
        differ in the following ways:

        * Non-reentrant checkpoint stops recomputation as soon as all needed
          intermediate activations have been recomputed. This feature is enabled
          by default, but can be disabled with :func:`set_checkpoint_early_stop`.
          Reentrant checkpoint always recomputes :attr:`function` in its
          entirety during the backward pass.

        * The reentrant variant does not record the autograd graph during the
          forward pass, as it runs with the forward pass under
          :func:`torch.no_grad`. The non-reentrant version does record the
          autograd graph, allowing one to perform backward on the graph within
          checkpointed regions.

        * The reentrant checkpoint only supports the
          :func:`torch.autograd.backward` API for the backward pass without its
          `inputs` argument, while the non-reentrant version supports all ways
          of performing the backward pass.

        * At least one input and output must have ``requires_grad=True`` for the
          reentrant variant. If this condition is unmet, the checkpointed part
          of the model will not have gradients. The non-reentrant version does
          not have this requirement.

        * The reentrant version does not consider tensors in nested structures
          (e.g., custom objects, lists, dicts, etc) as participating in
          autograd, while the non-reentrant version does.

        * The reentrant checkpoint does not support checkpointed regions with
          detached tensors from the computational graph, whereas the
          non-reentrant version does. For the reentrant variant, if the
          checkpointed segment contains tensors detached using ``detach()`` or
          with :func:`torch.no_grad`, the backward pass will raise an error.
          This is because ``checkpoint`` makes all the outputs require gradients
          and this causes issues when a tensor is defined to have no gradient in
          the model. To avoid this, detach the tensors outside of the
          ``checkpoint`` function.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint. Note that under torch.compile,
            this flag doesn't take effect and we always preserve RNG state.
            Default: ``True``
        use_reentrant(bool):
            specify whether to use the activation checkpoint variant that
            requires reentrant autograd. This parameter should be passed
            explicitly. In version 2.5 we will raise an exception if
            ``use_reentrant`` is not passed. If ``use_reentrant=False``,
            ``checkpoint`` will use an implementation that does not require
            reentrant autograd. This allows ``checkpoint`` to support additional
            functionality, such as working as expected with
            ``torch.autograd.grad`` and support for keyword arguments input into
            the checkpointed function.
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
            This argument is only supported if ``use_reentrant=False``.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. By default it is set to ``"default"`` which
            compares the shapes, dtypes, and devices of the recomputed tensors
            against those the saved tensors. To turn off this check, specify
            ``"none"``. Currently these are the only two supported values.
            Please open an issue if you would like to see more determinism
            checks. This argument is only supported if ``use_reentrant=False``,
            if ``use_reentrant=True``, the determinism check is always disabled.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation. This argument is only supported if
            ``use_reentrant=False``.
        args: tuple containing inputs to the :attr:`function`

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    if use_reentrant is None:
        warnings.warn(
            "torch.utils.checkpoint: the use_reentrant parameter should be "
            "passed explicitly. In version 2.5 we will raise an exception "
            "if use_reentrant is not passed. use_reentrant=False is "
            "recommended, but if you need to preserve the current default "
            "behavior, you can pass use_reentrant=True. Refer to docs for more "
            "details on the differences between the two variants.",
            stacklevel=2,
        )
        use_reentrant = True

    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs and use_reentrant:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    if use_reentrant:
        if context_fn is not noop_context_fn or debug is not False:
            raise ValueError(
                "Passing `context_fn` or `debug` is only supported when "
                "use_reentrant=False."
            )
        return UnslothCheckpointFunction.apply(function, preserve, *args)
    else:
        gen = _checkpoint_without_reentrant_generator(
            function, preserve, context_fn, determinism_check, debug, *args, **kwargs
        )
        # Runs pre-forward logic
        next(gen)
        ret = function(*args, **kwargs)
        # Runs post-forward logic
        try:
            next(gen)
        except StopIteration:
            return ret


def patch_unsloth_smart_gradient_checkpointing(dtype=None):
    # All Unsloth Zoo code licensed under LGPLv3
    if (
        torch.utils.checkpoint.CheckpointFunction.__name__
        != "UnslothCheckpointFunction"
    ):
        initialize_unsloth_gradient_checkpointing(dtype)
        torch.utils.checkpoint._old_CheckpointFunction = (
            torch.utils.checkpoint.CheckpointFunction
        )
        torch.utils.checkpoint.CheckpointFunction = UnslothCheckpointFunction

    if torch.utils.checkpoint.checkpoint.__name__ != "unsloth_checkpoint":
        torch.utils.checkpoint._old_checkpoint = torch.utils.checkpoint.checkpoint
        torch.utils.checkpoint.checkpoint = unsloth_checkpoint


def unpatch_unsloth_smart_gradient_checkpointing():
    # All Unsloth Zoo code licensed under LGPLv3
    if (
        torch.utils.checkpoint.CheckpointFunction.__name__
        == "UnslothCheckpointFunction"
    ) and hasattr(torch.utils.checkpoint, "_old_CheckpointFunction"):
        torch.utils.checkpoint.CheckpointFunction = (
            torch.utils.checkpoint._old_CheckpointFunction
        )
        global CPU_BUFFERS
        global GPU_BUFFERS
        for i in range(len(CPU_BUFFERS)):
            if hasattr(CPU_BUFFERS[i], "resize_"):
                CPU_BUFFERS[i].resize_(0)
            if type(CPU_BUFFERS) is list:
                CPU_BUFFERS[i] = None
        for i in range(len(GPU_BUFFERS)):
            if hasattr(GPU_BUFFERS[i], "resize_"):
                GPU_BUFFERS[i].resize_(0)
            if type(GPU_BUFFERS) is list:
                GPU_BUFFERS[i] = None
        CPU_BUFFERS = None
        GPU_BUFFERS = None

    if (torch.utils.checkpoint.checkpoint.__name__ == "unsloth_checkpoint") and hasattr(
        torch.utils.checkpoint, "_old_checkpoint"
    ):
        torch.utils.checkpoint.checkpoint = torch.utils.checkpoint._old_checkpoint


pass


# from https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/patching_utils.py
def patch_torch_compile(debug=False, O3=False, ignore_errors=True):
    # All Unsloth Zoo code licensed under LGPLv3
    assert type(debug) is bool
    assert type(O3) is bool
    import logging

    if debug:
        DEBUGGING = " with debugging"
        os.environ["TORCHDYNAMO_VERBOSE"] = "1"
        os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
        # os.environ["TORCH_LOGS"] = "dynamo,graph_breaks,recompiles,graph_code,aot_joint_graph,aot_graphs,compiled_autograd_verbose"
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        torch._logging.set_logs(
            dynamo=logging.WARN,
            inductor=logging.WARN,
            graph_breaks=True,
            recompiles=True,
            recompiles_verbose=True,
            compiled_autograd_verbose=False,  # Produces too much code
            aot_joint_graph=False,  # Produces too much code
            aot_graphs=False,  # Produces too much code
            perf_hints=True,  # Performance improvement hints
        )
        torch._dynamo.config.verbose = True
    else:
        DEBUGGING = ""
        os.environ.pop("TORCHDYNAMO_VERBOSE", None)
        os.environ.pop("TORCHINDUCTOR_COMPILE_THREADS", None)
        os.environ.pop("TORCHINDUCTOR_FORCE_DISABLE_CACHES", None)
        os.environ.pop("TORCH_LOGS", None)
        torch._logging.set_logs(all=logging.CRITICAL)
        torch._dynamo.config.verbose = False
    pass
    try:
        print(
            f"🦥 Unsloth Zoo will now patch everything{DEBUGGING} to make training faster!"
        )
    except:
        print(
            f"Unsloth Zoo will now patch everything{DEBUGGING} to make training faster!"
        )
    pass

    os.environ["UNSLOTH_PATCHED"] = "1"
    # See https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html
    # Caches kernel generations for faster restarts
    # https://dev-discuss.pytorch.org/t/impact-of-multithreading-and-local-caching-on-torch-compile/2498/3
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE"] = "1"
    os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)

    # Duplicate functions will cause hashing issues
    # os.environ["TORCHINDUCTOR_CACHE_DIR"] = UNSLOTH_COMPILE_LOCATION

    # https://github.com/sayakpaul/diffusers-torchao?tab=readme-ov-file#things-to-keep-in-mind-when-benchmarking
    os.environ["ENABLE_AOT_AUTOGRAD_CACHE"] = "1"

    # Torch compile arguments
    torch_compile_arguments = [
        f"config.debug = {debug}",
        "config.dce = True",
        "config.memory_planning = True",
        # Using 'combined' memory pool will cause re-compiles for dynamic shapres. We just re-use already allocated memory pools
        "config.memory_pool = 'none'",
        "config.efficient_conv_bn_eval_fx_passes = True",  # Reduces stability a little bit
        "config.dynamic_scale_rblock = True",  # Scale down RBLOCK for better occupancy
        # Disable reorder_for_compute_comm_overlap since it errors for non multi GPU systems
        # "config.reorder_for_compute_comm_overlap = True", # # enable reordering pass for increasing overlap between compute and communication
        f"config.max_autotune = {O3}",  # enable slow autotuning passes to select algorithms
        f"config.max_autotune_pointwise = {O3}",  # enable slow autotuning passes to select pointwise/reductions algorithms
        f"config.max_autotune_gemm = False",  # GEMM is unnecessary
        "config.max_autotune_gemm_backends = 'ATEN,TRITON,CPP'",  # Not much faster
        "config.autotune_fallback_to_aten = True",  # Fallback to ATEN backend
        "config.autotune_multi_device = True",  # If autotuning in subprocess, whether to use multiple devices
        f"config.coordinate_descent_tuning = {O3}",
        f"config.aggressive_fusion = {O3}",  # Careful changes results!
        # [TODO] COMBO KERNELS makes everything slower!
        # "config.combo_kernels = True", # Experimental - enable the combo kernel that combines data-independent kernels
        # "config.combo_kernel_foreach_dynamic_shapes = True",
        "config.freezing = False",  # Freezes weights --> ** only useful for inference **
        # f"config.triton.multi_kernel = {O3}", # use tuning to pick between different subkernels
        "config.cuda.enable_cuda_lto = True",
        "config.cuda.use_fast_math = True",
        f"config.cuda.compile_opt_level = {'-O2' if O3 else '-O1'}",
        # See torch.compile, the missing manual
        # https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8
        # f"config.emulate_precision_casts = {not debug}", # Force X.to(f32).to(f16) instead of X.to(f16)
        # when setting to not debug aka True, we get errors on torch2.6
        # TypeError: ValueRangeAnalysis.to_dtype() got an unexpected keyword argument 'use_compute_types'
        # this keyword exists in torch2.7.0 but not in torch2.6.0 so set to False until torch2.6.0 is deprecated.
    ]
    # Torch dynamo arguments
    torch_dynamo_arguments = [
        "config.accumulated_cache_size_limit = 1024",  # Bump up a bit from 256
        f"config.suppress_errors = {not debug and ignore_errors}",  # Supress errors for now
        f"config.do_not_emit_runtime_asserts = {not debug}",
        "config.inline_inbuilt_nn_modules = True",  # Torch 2.5 Regional recompilation
        "config.numpy_default_float = 'float32'",
        # FAILS for Gemma!
        "config.compiled_autograd = False",  # New Torch 2.4 feature which can compile backwards passes
        # https://pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html
        # [NOTE] recompile_limit and cache_size_limit are equivalent!
        "config.recompile_limit = 1024",  # Increase recompile amounts to 1024 - then will do eager
        "config.cache_size_limit = 1024",  # Flex Attention
        # f"config.fail_on_recompile_limit_hit = {not debug and ignore_errors}", # Ignore recompiles CANNOT be used in tandem with suppress_errors
        "config.allow_unspec_int_on_nn_module = True",  # Integers in modules will auto wrap torch.tensor(self.vocab_size)
        f"config.optimize_ddp = {not debug}",  # Optimizes DDP, but can error out so disable on debug
        # Captures .item() for eg
        # n_chunks = int(torch.ceil((torch.tensor(vocab_size) / 262144) * 8))
        "config.capture_scalar_outputs = True",
        # Capture torch.arange(...), torch.zeros(...)
        "config.capture_dynamic_output_shape_ops = True",
    ]
    if not debug and ignore_errors:
        # Have to explicitly set it!
        torch._dynamo.config.suppress_errors = True
    pass
    import torch._inductor.config as config

    for _try_compile_argument in torch_compile_arguments:
        try:
            exec(_try_compile_argument)
        except:
            pass
    pass
    import torch._dynamo.config as config

    for _try_dynamo_argument in torch_dynamo_arguments:
        try:
            exec(_try_dynamo_argument)
        except:
            pass
    pass


pass


def patch_compiled_autograd():
    # Fixes double compilation of functions during gradient checkpointing
    # See https://github.com/pytorch/pytorch/issues/135298
    # All Unsloth Zoo code licensed under LGPLv3
    import inspect, re

    # From https://github.com/pytorch/pytorch/pull/135795/files
    import torch._dynamo.compiled_autograd

    fx = torch._dynamo.compiled_autograd.AutogradCompilerInstance.end_capture
    if fx.__name__ == "unsloth_end_capture":
        return
    source = inspect.getsource(fx)
    if "with disable()" in source:
        return
    spaces = source.find("def")
    source = source.split("\n")
    source = "\n".join(x[spaces:] for x in source)
    old = "return compiled_fn(inputs, sizes, scalars, hooks)"
    match = re.search(r"\n([ ]{1,})return compiled_fn", source)
    n = len(match.group(1)) if match else 0
    source = source.replace(old, f"with disable():\n{' ' * (n + 4)}{old}")
    source = source.replace("def end_capture", "def unsloth_end_capture", 1)

    # Import items to make the function executable
    all_items = dir(torch._dynamo.compiled_autograd)
    good_items = [x for x in all_items if x in source]
    exec(
        "from torch._dynamo.compiled_autograd import ("
        + ", ".join(x for x in good_items)
        + ")",
        globals(),
    )
    exec(source, globals())
    torch._dynamo.compiled_autograd.AutogradCompilerInstance.end_capture = (
        unsloth_end_capture
    )

    # From https://github.com/pytorch/pytorch/pull/135795/files
    try:
        import torch._dynamo.variables.misc

        fx = torch._dynamo.variables.misc.AutogradEngineVariable.call_method
    except:
        return
    if fx.__name__ == "unsloth_call_method":
        return
    source = inspect.getsource(fx)
    if "in_compiled_autograd_region" in source:
        return
    spaces = source.find("def")
    source = source.split("\n")
    source = "\n".join(x[spaces:] for x in source)
    source = source.replace(
        "torch._dynamo.compiled_autograd.compiled_autograd_enabled",
        "torch._dynamo.compiled_autograd.in_compiled_autograd_region",
        1,
    )
    source = source.replace("def call_method", "def unsloth_call_method", 1)

    # Import items to make the function executable
    all_items = dir(torch._dynamo.variables.misc)
    good_items = [x for x in all_items if x in source]
    exec(
        "from torch._dynamo.variables.misc import ("
        + ", ".join(x for x in good_items)
        + ")",
        globals(),
    )
    exec(source, globals())
    torch._dynamo.variables.misc.AutogradEngineVariable.call_method = (
        unsloth_call_method
    )
    return


pass
