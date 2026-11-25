"""Capacity and training-loop parity checks, without DEV selection."""
import ast
import torch
from vp_1103_common import WORK,atomic_json,now,sha,provenance
from vp_1103_unet_matched_model import MatchedUNet,count_formula,WIDTHS

class NormalizeLoop(ast.NodeTransformer):
    def visit_Constant(self,node):
        if node.value in ('recovp_formal_dev','unet_matched58943'):node.value='RUN_ID'
        elif node.value in ('ReCoVP-Base','UNet-Matched58943'):node.value='ARM'
        return node
    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id in ('make_recovp','make_unet'):
            return ast.Call(func=ast.Name(id='MODEL_FACTORY',ctx=ast.Load()),args=[],keywords=[])
        return self.generic_visit(node)

def run():
    torch.set_num_threads(1);torch.manual_seed(17)
    model=MatchedUNet();x=torch.randn(2,2,1800);out=model(x)
    out.square().mean().backward()
    checks={name:bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0) for name,p in model.named_parameters()}
    registered=sum(p.numel() for p in model.parameters());active=sum(p.numel() for p in model.parameters() if p.grad is not None)
    assert registered==active==count_formula(*WIDTHS)==58943
    assert all(checks.values()) and torch.isfinite(out).all() and out.shape==x.shape
    def loop(name):
        tree=ast.parse((WORK/'tools'/name).read_text(encoding='utf8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        return ast.dump(NormalizeLoop().visit(fn),include_attributes=False)
    assert loop('vp_1103_recovp_gpu_train.py')==loop('vp_1103_unet_matched_train.py')
    result={**provenance(__file__,WORK/'tools/vp_1103_unet_matched_model.py',WORK/'tools/vp_1103_unet_matched_train.py'),
        'utc':now(),'status':'PASS','registered_real_parameters':registered,'active_real_parameters':active,'widths':list(WIDTHS),
        'no_padding_or_disconnected_parameters':True,'all_parameter_tensors_finite_nonzero_gradient':checks,
        'training_loop_ast_equivalent_after_model_factory_and_run_identity_normalization':True,
        'selection':'Widths selected by count only; no DEV read by this check',
        'matching_reference':'ReCoVP-NoEquiv 58943 active real parameters'}
    atomic_json(WORK/'evidence/outputs/unet_matched58943/PREFLIGHT.json',result)
    print({'status':'PASS','active_real_parameters':active,'training_loop_parity':'PASS'})
    return result

if __name__=='__main__':run()
