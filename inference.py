import sys

from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

from util import get_base_data_as_sample

samples = get_base_data_as_sample()
model = XMLBIFReader("model.xml").get_model()

# 3. Causal Inference

infer_non_adjust = VariableElimination(model)
# print(infer_non_adjust.query(variables=["time_SLO"]))
# print(infer_non_adjust.query(variables=["success"],
#                              evidence={'within_time': True, 'fps': 60}))

bitrate_list = model.get_cpds("bitrate").__getattribute__("state_names")["bitrate"]
bitrate_comparison = []

print(infer_non_adjust.query(variables=["time_SLO"]))

for br in bitrate_list:
    sr = infer_non_adjust.query(variables=["distance_SLO", "time_SLO"], evidence={'bitrate': br}).values[1][1]
    if sr > 0.7:
        cons = samples[samples['bitrate'] == br]['consumption'].mean()
        bitrate_comparison.append((br, sr,
                                   samples[samples['bitrate'] == br]['pixel'].iloc[0],
                                   samples[samples['bitrate'] == br]['fps'].iloc[0],
                                   cons))

for (br, sr, pixel, fps, cons) in bitrate_comparison:
    print(pixel, fps, sr, cons)

sys.exit()
