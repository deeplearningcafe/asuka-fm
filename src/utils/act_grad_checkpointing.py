import torch
import threading
import queue
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
    def __init__(self, model):
        self.model = model
        self.grad_buffers = {}

        # Stream for async memory transfers
        self.copy_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        # Queue and worker thread for background CPU accumulation
        self.accumulation_queue = queue.Queue()
        self.worker_thread = threading.Thread(
            target=self._accumulation_worker, daemon=True
        )
        self.worker_thread.start()
        # The problem with the grad accum is that it's completely cpu dependant
        # if other process is using 100% cpu then this grad accum is ultra slow

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.grad_buffers[param] = torch.zeros(
                    param.shape, dtype=torch.float32, device="cpu", pin_memory=True
                )

                param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _accumulation_worker(self):
        """
        Background thread that waits for GPU transfers to finish
        and then safely adds the gradients to the CPU buffers.
        """
        while True:
            task = self.accumulation_queue.get()
            if task is None:
                self.accumulation_queue.task_done()
                break

            grad_cpu, param, event = task

            if event is not None:
                # CPU thread waits for the async CUDA transfer to finish.
                event.synchronize()

            self.grad_buffers[param].add_(grad_cpu)

            # Mark task as complete so we can join() later
            self.accumulation_queue.task_done()

    def _make_hook(self, param):
        def hook_fn(p):
            if p.grad is not None:
                if self.copy_stream:
                    # 1. Make copy stream wait for the compute stream to finish the gradient
                    self.copy_stream.wait_stream(torch.cuda.current_stream())

                    with torch.cuda.stream(self.copy_stream):
                        grad_cpu = p.grad.to("cpu", non_blocking=True).float()

                        # Record an event to track when the copy completes
                        event = torch.cuda.Event()
                        event.record(self.copy_stream)

                    # Prevent caching allocator from freeing p.grad before copy completes
                    p.grad.record_stream(self.copy_stream)

                    # Queue the CPU addition for the background thread
                    self.accumulation_queue.put((grad_cpu, p, event))
                else:
                    grad_cpu = p.grad.to("cpu").float()
                    self.accumulation_queue.put((grad_cpu, p, None))

                p.grad = None

        return hook_fn

    def finalize_and_step(self, optimizer, scaler=None, max_norm=1.0):
        # Wait for all background accumulations to finish before stepping!
        self.accumulation_queue.join()

        for param, cpu_grad in self.grad_buffers.items():
            if param.grad is None:
                # Use zeros_like instead of empty_like to prevent uninitialized memory (NaNs)
                param.grad = torch.zeros_like(param)

            # Use non_blocking=False. Since we immediately zero the CPU buffer below,
            # an async copy (non_blocking=True) creates a race condition where cpu_grad is
            # zeroed before the DMA transfer completes, sending zeroes to the GPU.
            param.grad.copy_(cpu_grad, non_blocking=False)

            cpu_grad.zero_()

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            for param in self.grad_buffers.keys():
                if param.grad is not None:
                    torch.distributed.all_reduce(
                        param.grad, op=torch.distributed.ReduceOp.SUM
                    )
                    param.grad.div_(world_size)

        # Calculate Norm & Coef on GPU
        total_norm = torch.nn.utils.clip_grad_norm_(self.grad_buffers.keys(), max_norm)

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Clear GPU VRAM
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

    # Allocate buffers to how many GPUs
    n_gpus = (
        torch.cuda.device_count() if DEVICE_TYPE == "cuda" else torch.xpu.device_count()
    )
    GPU_BUFFERS = tuple(
        [
            torch.empty(2 * 256 * 2048, dtype=dtype, device=f"{DEVICE_TYPE}:{i}")
            for i in range(n_gpus)
        ]
    )

    BACKWARD_PASS = True
    EXTRA_STREAMS = tuple(
        [
            torch.cuda.Stream() if DEVICE_TYPE == "cuda" else torch.xpu.Stream()
            for i in range(n_gpus)
        ]
    )
    if DEVICE_TYPE == "cuda":
        MAIN_STREAMS = tuple(
            [
                torch.cuda.default_stream(torch.device(f"cuda:{i}"))
                for i in range(n_gpus)
            ]
        )
    elif DEVICE_TYPE == "xpu":
        MAIN_STREAMS = tuple(
            [torch.xpu.current_stream(torch.device(f"xpu:{i}")) for i in range(n_gpus)]
        )

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
