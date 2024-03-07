from math import asinh, pi, exp, log10
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optical_rl_gym.envs.prmsa_env import PRMSAEnv
from optical_rl_gym.utils import Service, Span, Link

def calculate_osnr(env: 'PRMSAEnv', current_service: Service):
    beta_2 = -21.3e-27  # group velocity dispersion (s^2/m)
    gamma = 1.3e-3  # nonlinear parameter 1/(W.m)
    h_plank = 6.626e-34  # Planck's constant (J s)
    acc_gsnr = 0
    l_eff_a = 0
    l_eff = 0
    phi = 0
    sum_phi = 0
    power_ase = 0
    power_nli_span = 0
    phi_modulation_format = np.array((1, 1, 2/3, 17/25, 69/100, 13/21))

    for link in current_service.path.links:
        for span in env.topology[link.node1][link.node2]["link"].spans:
            l_eff_a = 1 / (2 * span.attenuation_normalized)
            l_eff = (1 - np.exp(-2 * span.attenuation_normalized * span.length * 1e3)) / (2 * span.attenuation_normalized)

            sum_phi = asinh(
                pi ** 2 * abs(beta_2) * (current_service.bandwidth) ** 2 / (4 * span.attenuation_normalized)
            )

            for service in env.topology[link.node1][link.node2]["running_services"]:
                if service.service_id != current_service.service_id:
                    phi = (
                        asinh(
                            pi ** 2 * abs(beta_2) * l_eff_a * service.bandwidth *
                            (service.center_frequency - current_service.center_frequency + (service.bandwidth / 2))
                        ) - asinh(
                            pi ** 2 * abs(beta_2) * l_eff_a * service.bandwidth *
                            (service.center_frequency - current_service.center_frequency - (service.bandwidth / 2))
                        )
                    ) - (
                        phi_modulation_format[service.current_modulation.spectral_efficiency - 1] *
                        (service.bandwidth / abs(service.center_frequency - current_service.center_frequency)) *
                        5 / 3 * (l_eff / (span.length * 1e3))
                    )
                sum_phi += phi

            power_nli_span = (current_service.launch_power / (current_service.bandwidth)) ** 3 * \
                             (8 / (27 * pi * abs(beta_2))) * gamma ** 2 * l_eff * sum_phi * (current_service.bandwidth)
            power_ase = current_service.bandwidth * h_plank * current_service.center_frequency * \
                        (exp(2 * span.attenuation_normalized * span.length * 1e3) - 1) * span.noise_figure_normalized

            acc_gsnr += 1 / (current_service.launch_power / (power_ase + power_nli_span))

    gsnr = 10 * np.log10(1 / acc_gsnr)
    return gsnr


def calculate_osnr_default_attenuation(env: 'PRMSAEnv', current_service: Service, attenuation_normalized: float, noise_figure_normalized: float):
    beta_2 = -21.3e-27  # group velocity dispersion (s^2/m)
    gamma = 1.3e-3  # nonlinear parameter 1/(W.m)
    h_plank = 6.626e-34  # Planck's constant (J s)
    acc_gsnr = 0
    l_eff_a = 0
    l_eff = 0
    phi = 0
    sum_phi = 0
    phi_modulation_format = np.array((1, 1, 2/3, 17/25, 69/100, 13/21))

    for link in current_service.path.links:
        for span in link.spans:
            l_eff_a = 1 / (2 * attenuation_normalized)
            l_eff = (1 - np.exp(-2 * attenuation_normalized * span.length * 1e3)) / (2 * attenuation_normalized)

            sum_phi = asinh(
                pi ** 2 * abs(beta_2) * (current_service.bandwidth) ** 2 / (4 * attenuation_normalized)
            )

            for service in env.topology[link.node1][link.node2]["running_services"]:
                if service.center_frequency - current_service.center_frequency == 0:
                    print(service)
                    print(current_service)
                if service.service_id != current_service.service_id:
                    phi = (
                        asinh(
                            pi ** 2 * abs(beta_2) * l_eff_a * service.bandwidth *
                            (service.center_frequency - current_service.center_frequency + (service.bandwidth / 2))
                        ) - asinh(
                            pi ** 2 * abs(beta_2) * l_eff_a * service.bandwidth *
                            (service.center_frequency - current_service.center_frequency - (service.bandwidth / 2))
                        )
                    ) - (
                        phi_modulation_format[service.current_modulation.spectral_efficiency - 1] *
                        (service.bandwidth / abs(service.center_frequency - current_service.center_frequency)) *
                        5 / 3 * (l_eff / (span.length * 1e3))
                    )
                sum_phi += phi

            power_nli_span = (env.current_service.launch_power / (env.current_service.bandwidth)) ** 3 * \
                             (8 / (27 * pi * abs(beta_2))) * gamma ** 2 * l_eff * sum_phi * (env.current_service.bandwidth)
            power_ase = env.current_service.bandwidth * h_plank * env.current_service.center_frequency * \
                        (exp(2 * attenuation_normalized * span.length * 1e3) - 1) * noise_figure_normalized

            acc_gsnr += 1 / (env.current_service.launch_power / (power_ase + power_nli_span))

    gsnr = 10 * np.log10(1 / acc_gsnr)
    return gsnr
