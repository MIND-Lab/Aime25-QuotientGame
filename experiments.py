#from shap import leave_one_patient_out as s_leave_one_patient_out
from grad_cam import leave_one_patient_out as gc_leave_one_patient_out
from lime import leave_one_patient_out as l_leave_one_patient_out
from shap import leave_one_patient_out as s_leave_one_patient_out
from qg_experiment import leave_one_patient_out as qg_leave_one_patient_out
from DatasetClass import Dataset

dictionary = {'PD':0, 'AD':1, 'CTRL':2}
ds = Dataset.load_file('pkl', 'dataset/dataset_pd_ad.pkl','', dictionary)

gc_leave_one_patient_out("pd-ad/weights/", ds, "pd-ad/raw/grad_cam/")
l_leave_one_patient_out("pd-ad/weights/", ds, "pd-ad/raw/lime/")
s_leave_one_patient_out("pd-ad/weights/", ds, "pd-ad/raw/shap/")
qg_leave_one_patient_out("pd-ad/weights/", ds, "pd-ad/raw/qg/")