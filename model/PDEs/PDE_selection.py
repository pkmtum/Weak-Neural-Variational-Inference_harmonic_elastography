from utils.function_load_from_list import load_from_list

def PDE_selection(PDE_kind, PDE_options):
    prior_classes = {
        "Hook": "model.PDEs.class_Hook_analytical.HookAnalytical",
        "NeoHook": "model.PDEs.class_NeoHook_analytical.NeoHookAnalytical",
        "HarmonicHook": "model.PDEs.class_HarmonicHook_analytical.HarmonicHook",
    }
    return load_from_list(prior_classes, PDE_kind, PDE_options)