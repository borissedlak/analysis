import sys

from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

from util import get_prepared_base_samples

samples = get_prepared_base_samples()
model = XMLBIFReader("model.xml").get_model()


# 3. Causal Inference


def get_config_for_model(distance_slo, time_slo, success_slo, network_slo, gpu):
    var_el = VariableElimination(model)

    mode_list = model.get_cpds("config").__getattribute__("state_names")["config"]
    bitrate_list = model.get_cpds("bitrate").__getattribute__("state_names")["bitrate"]
    config_line = []

    for br in bitrate_list:
        for mode in mode_list:
            distance = var_el.query(variables=[distance_slo], evidence={'bitrate': br, 'config': mode, 'GPU': gpu}).values[1]
            time = var_el.query(variables=["time_SLO"], evidence={'bitrate': br, 'config': mode, 'GPU': gpu}).values[1]
            transformed = var_el.query(variables=["transformed"], evidence={'bitrate': br, 'config': mode, 'GPU': gpu}).values[1]

            if distance >= 0.95 and time >= time_slo and transformed >= success_slo and int(br) <= network_slo:
                cons = samples[samples['bitrate'] == br]['consumption'].mean()  # this must be evaluated live
                config_line.append((br, distance, time, transformed, mode,
                                           samples[samples['bitrate'] == br]['pixel'].iloc[0],
                                           samples[samples['bitrate'] == br]['fps'].iloc[0],
                                           cons))

    config_line = sorted(config_line, key=lambda x: x[7])
    for (br, distance, time, transformed, mode, pixel, fps, cons) in config_line:
        print(pixel, fps, mode, distance, time, transformed, cons)


print("Pixel, FPS, Config, Distance, Time, Success, Consumption")
print("--------------Scenario A --------------------")
get_config_for_model(distance_slo="distance_SLO_hard", time_slo=0.95, success_slo=0.90, network_slo=(409920*20), gpu='False')

print("\n--------------Scenario B --------------------")
get_config_for_model(distance_slo="distance_SLO_easy", time_slo=0.75, success_slo=0.98, network_slo=(230400*16), gpu='True')  # Scenario B

sys.exit()
