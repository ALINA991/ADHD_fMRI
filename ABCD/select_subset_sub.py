from pathlib import Path
import pandas as pd
import numpy as np

data_root = Path("/Volumes/Samsung_T5/MIT/abcd/release_05/abcd-data-release-5.1/core")
nc_section = "neurocognition"
data_path = Path(data_root, nc_section)
ddis_name = "nc_y_ddis.csv"
ddis = pd.read_csv(Path(data_path, ddis_name))

ddis_ = ddis.query("eventname == '1_year_follow_up_y_arm_1'").head(200)
sub_ids = np.array(ddis_['src_subject_id'], dtype=str)
np.save(Path(data_root,'subset_subjects_ids.npy'), sub_ids)
