from triton.language import core

@core.extern
def gdc_wait(_semantic=None):
    core.inline_asm_elementwise("griddepcontrol.wait; // dummy $0", "=r", [], dtype=core.int32, is_pure=False, pack=1,
                                _semantic=_semantic)

@core.extern
def gdc_launch_dependents(_semantic=None):
    core.inline_asm_elementwise("griddepcontrol.launch_dependents; // dummy $0", "=r", [], dtype=core.int32,
                                is_pure=False, pack=1, _semantic=_semantic)
