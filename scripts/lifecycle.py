"""Explicit SDK lifecycle. Imported safely; changes state only when called by --execute."""
import time
from pathlib import Path


def loader_spec(model):
    relative = model['sdk_key']
    root = Path.home()/'.lmstudio/models'
    file = (root/relative).resolve()
    if not file.is_relative_to(root.resolve()) or not file.is_file():
        raise ValueError(f'Local GGUF not found under LM Studio models: {file}')
    if not file.name.lower().endswith('.gguf') or model['quant'].lower() not in file.name.lower():
        raise ValueError('GGUF filename does not match the requested quantization')
    return {'sdk_key': relative, 'file': str(file), 'size_bytes': file.stat().st_size,
            'identifier': model['identifier'], 'expected_quantization': model['quant']}


def snapshot(read, timeout=30):
    """Retry only the known transient CLI model-inventory race."""
    stop = time.monotonic()+timeout
    while True:
        try:
            return read()
        except RuntimeError as exc:
            if 'Model does not exist' not in str(exc) or time.monotonic() >= stop:
                raise
            time.sleep(.5)


def unload_owned(model, read, verify, client, emit):
    current = snapshot(read)
    if not current:
        return
    verify(current, model)  # Do not unload another model or another user's instance.
    if current[0].get('queued', 0) or current[0].get('status') not in (None, 'idle'):
        raise RuntimeError('Model is busy; automatic unload refused')
    emit('auto_unload_start', identifier=model['identifier'])
    client.llm.unload(model['identifier'])
    stop = time.monotonic()+30
    while snapshot(read):
        if time.monotonic() >= stop:
            raise TimeoutError('Model did not unload within 30 seconds')
        time.sleep(.5)
    emit('auto_unload_end', identifier=model['identifier'])


def sdk_load(client, key, identifier, config):
    # SDK 1.5 lacks public AutoFit/parallel/speculation fields; use its load channel with
    # upstream KV keys. Kept here, version-pinned and validated after loading.
    from lmstudio.json_api import LoadModelEndpoint
    api = client.llm
    endpoint = LoadModelEndpoint(key, identifier, None,
                                api._API_TYPES.REQUEST_NEW_INSTANCE,
                                api._API_TYPES.MODEL_LOAD_CONFIG, config, None)
    endpoint.creation_params['loadConfigStack']['layers'][-1]['config']['fields'].extend([
            # Required by current GGUF runtime: otherwise AutoFit overrides contextLength.
            {'key': 'llm.load.llama.autoFit', 'value': False},
            {'key': 'llm.load.numParallelSessions', 'value': 1},
            {'key': 'llm.load.speculativeDecoding.draftMtp', 'value': False},
            {'key': 'llm.load.speculativeDecoding.draftSimple', 'value': False},
        ])
    with api._create_channel(endpoint) as channel:
        result = channel.wait_for_result()
        return api._create_handle(result.identifier)


def load_exact(model, window, client, read, verify, emit):
    spec = loader_spec(model)
    if snapshot(read):
        raise RuntimeError('Another model is resident; refusing concurrent loading')
    config = {'contextLength': window, 'gpu': {'ratio': 1.0}, 'flashAttention': True}
    emit('auto_load_start', loader=spec, requested_load_config=config)
    # Never fall back to the base model key: it previously selected Q4 instead of Q8.
    handle = sdk_load(client, model['key'], model['identifier'], config)
    current = snapshot(read)
    verify(current, model)
    if current[0].get('parallel') != 1:
        raise RuntimeError('Loaded instance must have parallel=1')
    if handle.get_context_length() != window:
        raise RuntimeError(f'SDK context {handle.get_context_length()} != requested {window}; metadata={current}')
    expected_size = model.get('expected_bundle_size_bytes')
    if expected_size and current[0].get('sizeBytes') != expected_size:
        raise RuntimeError('Loaded model bundle size differs from previous runs; possible missing vision projector')
    actual = handle.get_load_config().to_dict()
    if actual.get('gpu', {}).get('ratio') != 1.0 or actual.get('flashAttention') is not True:
        raise RuntimeError('GPU/Flash Attention settings differ from the measured baseline')
    emit('auto_load_verified', model=current[0], load_config=actual)
    return handle, current


def memory_root_pid():
    import psutil
    candidates = []
    for p in psutil.process_iter(['pid','name','exe']):
        info = p.info
        if (info['name'] or '').lower() in ('bionic','lm studio','lm-studio','llmster'):
            candidates.append(info['pid'])
    if len(candidates) != 1:
        raise ValueError(f'Cannot select one memory root automatically: {candidates}; pass --memory-pid')
    return candidates[0]
