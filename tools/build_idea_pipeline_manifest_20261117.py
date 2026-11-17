"""Export an honest task DAG from dated evidence; never launches training."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evidence/outputs'


def main():
    snapshot = json.loads((OUT/'IDEA_PIPELINE_STATUS_20261117.json').read_text('utf-8'))
    tasks = [
        {'id': 'D0', 'name': 'Residual/failure and retrospective JNR diagnosis',
         'status': 'COMPLETE_DIAGNOSTIC', 'deployed': False, 'depends_on': [],
         'evidence': ['RESIDUAL_FAILURE_DIAGNOSTIC_REPORT_20261117.md'],
         'claim_boundary': 'Association and retrospective findings; no new-model efficacy claim'},
        {'id': 'D1', 'name': 'Common phase/scale diagnostic; fixed 16 DEV parents',
         'status': 'COMPLETE_DIAGNOSTIC', 'deployed': False, 'depends_on': [],
         'evidence': ['phase_scale_diagnostic_20261117_v1/SUMMARY.json'],
         'claim_boundary': 'Not a delay test, full DEV detection analysis, or teacher evaluation'},
    ]
    names = {'tail': ('T', 'unet_mf_tail_screen_20261117_v2'),
             'norm': ('N', 'unet_noise_norm_screen_20261117_v1'),
             'dilation': ('L', 'unet_dilation_screen_20261117_v1'),
             'frequency': ('F', 'unet_frequency_screen_20261117_v1')}
    for factor, (identity, run) in names.items():
        job = next(x for x in snapshot['jobs'] if x['run'] == run)
        tasks.append({
            'id': identity, 'name': factor+' independent three-seed screen', 'run': run,
            'status': job['status']['stage'], 'deployed': True, 'physical_gpu': job['physical_gpu'],
            'depends_on': ['T'] if factor == 'dilation' else ['N'] if factor == 'frequency' else [],
            'dependency_type': 'Training plus paired DEV completion for device scheduling; prerequisite need not pass efficacy gate',
            'saved_epochs': {str(s['seed']): s['saved_epoch'] for s in job['seeds']},
            'seeds': [17,29,41], 'epochs_per_seed': 256, 'updates_per_seed': 12288,
            'includes_automatic_paired_dev_evaluation': True,
            'automatic_completion_evidence': run+'/PIPELINE_STATUS.json',
            'scope': 'Development only; no selection of best epoch or best seed',
            'eta_training_hours_only': job['estimated_training_hours_remaining_current_speed'],
            'queue_binding_sha256': job['queue_binding_sha256'],
            'config_sha256': job['config_sha256'], 'source_drift': job['source_drift']})
    planned = [
        ('D2','Delay/boundary, phase-failure association, peak denominator sensitivity and common-DEV teacher audit',
         ['D0','D1'],'PLANNED','Diagnostic protocols and teacher/data identity audit before execution'),
        ('S','Four-factor paired DEV review and original-grid main/extreme/JNR stratification',
         ['T','N','L','F'],'PLANNED','All four fixed endpoint summaries; keep all seed-level effects and strata counts'),
        ('C0','Mean-PC and hard-parent control experiments',
         ['T'],'CONDITIONAL','Tail passes gate; prefreeze controls and budget before reading their outcomes'),
        ('C1','Architecture-by-loss four cells and minimal combinations',
         ['S'],'CONDITIONAL','Only individually eligible factors; do not automatically stack factors'),
        ('R0','Equal-augmentation U-Net / ordinary complex / minimal equivariant control',
         ['D2'],'CONDITIONAL','Transformation inconsistency linked to relevant failures; stop if augmentation explains benefit'),
        ('R1','Truth / output KD / correction KD / selective relational KD',
         ['D2'],'CONDITIONAL','Teacher has common-DEV background advantage and clean data-role provenance'),
        ('M','Coherent-jammer generator and same-eight-pulse-input temporal/2D U-Net controls',
         ['D2'],'DEFERRED','No multi-pulse architecture training until generator/coherence and information fairness are verified'),
        ('X','Training-parent scale, shared-device efficiency, selected mismatch checks',
         ['C1'],'CONDITIONAL','Stable candidate; optional R0/R1/M only if their branch is activated and passes'),
        ('E8','Final candidate and key U-Net controls; eight common seeds to epoch 256',
         ['X'],'CONDITIONAL','Freeze candidate; reuse completed seeds only if full run identity matches'),
        ('H0','Freeze independent calibration protocol, data roles and sample sizes',
         ['E8'],'PLANNED','No threshold/architecture tuning on confirmation; teacher/student train isolation'),
        ('Q','New independent confirmation and parent/seed uncertainty analysis',
         ['H0'],'PLANNED','Frozen deployable calibration thresholds; E and J+E false alarms reported separately'),
        ('P','Nearest-prior full-text comparison, manuscript claims and reproducible release',
         ['Q'],'PLANNED','Evidence-grounded novelty, efficiency, limitations and reporting'),
    ]
    for identity, name, dependencies, status, gate in planned:
        tasks.append({'id':identity,'name':name,'depends_on':dependencies,
                      'status':status,'deployed':False,'entry_gate':gate,
                      'eta_training_hours_only':None})
    result = {
        'schema':'isrj2-research-pipeline-v1', 'snapshot_utc':snapshot['checked_utc'],
        'status_evidence_sha256':hashlib.sha256((OUT/'IDEA_PIPELINE_STATUS_20261117.json').read_bytes()).hexdigest(),
        'report':'IDEA_TO_EXPERIMENT_PIPELINE_REVIEW_20261117.md',
        'automatic_execution_scope':['T','N','L','F'],
        'not_a_controller':'This JSON records plans; PLANNED/CONDITIONAL/DEFERRED tasks are not launched jobs.',
        'development_gate':{'median_seed_delta_frame_pfa_jammer':'< 0',
                            'each_seed_delta_pd':'>= -0.02',
                            'mean_seed_delta_damage':'<= +0.005',
                            'interpretation':'Resource screening only, not significance or confirmed noninferiority'},
        'constraints':{'server':'rsdr-b','root':'D:/code/Xin/ISRJ_222',
                       'local_training':False,'remote_for_push':'paper2',
                       'all_epoch_recovery':True,'milestones_every_epochs':32,
                       'upload_training_data_weights_raw_arrays_full_logs':False},
        'tasks':tasks}
    target=OUT/'IDEA_EXPERIMENT_PIPELINE_20261117.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'tasks':len(tasks),'deployed':[x['id'] for x in tasks if x['deployed']],
                      'snapshot_utc':result['snapshot_utc']},ensure_ascii=False))


if __name__=='__main__': main()
