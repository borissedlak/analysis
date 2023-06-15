import sys

from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

from util import get_prepared_base_samples

samples = get_prepared_base_samples()
model = XMLBIFReader("model.xml").get_model()

# 3. Causal Inference

var_el = VariableElimination(model)

bitrate_list = model.get_cpds("bitrate").__getattribute__("state_names")["bitrate"]
bitrate_comparison = []

for br in bitrate_list:
    sr = var_el.query(variables=["distance_SLO", "time_SLO"], evidence={'bitrate': br}).values[1][1]
    if sr > 0.78:
        cons = samples[samples['bitrate'] == br]['consumption'].mean()
        bitrate_comparison.append((br, sr,
                                   samples[samples['bitrate'] == br]['pixel'].iloc[0],
                                   samples[samples['bitrate'] == br]['fps'].iloc[0],
                                   cons))

for (br, sr, pixel, fps, cons) in bitrate_comparison:
    print(pixel, fps, sr, cons)

sys.exit()
