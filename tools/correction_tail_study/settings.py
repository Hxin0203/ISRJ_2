"""Server-only original-backbone loss study; fixed arm and physical device."""
import os

ARM = os.environ.get('ISRJ2_CORRECTION_ARM')
if ARM not in ('original', 'tail'):
    raise RuntimeError('ISRJ2_CORRECTION_ARM must select original or tail')
GPU_INDEX = '1'
GPU_UUID = 'GPU-bb60dd3a-738e-2398-908f-7f77ac05549b'
CONTROL_RUN = 'correction_original_control_20261117_v1'
TAIL_RUN = 'correction_mf_tail_20261117_v1'
RUN = CONTROL_RUN if ARM == 'original' else TAIL_RUN
DEPENDENCY = 'unet_mf_tail_screen_20261117_v2' if ARM == 'original' else CONTROL_RUN
PAUSE_OWNER = 'ISRJ2_CORRECTION_TAIL_STUDY_20261117_V1'
DILATION_RUN = 'unet_dilation_screen_20261117_v1'
