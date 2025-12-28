"""
Temperature sensor modules for NI DAQ hardware.

This module provides temperature sensor classes for National Instruments
data acquisition hardware.
"""

import logging

import nidaqmx
import numpy as np

logger = logging.getLogger(__name__)

# Registry of available temperature sensor classes
class_list = {
    "NITempSensor": "NITempSensor",
}


class NITempSensor:
    """NI DAQ temperature sensor using thermocouple input."""

    def __init__(self, device, device_params, thrmcpl_type):
        """
        Initialize NI temperature sensor.

        Parameters
        ----------
        device : nidaqmx.system.Device
            NI device object with name, product_category, product_type.
        device_params : dict
            Configuration with keys: nsamples, thrmcpl_chan, cjc_source.
        thrmcpl_type : str
            Thermocouple type (e.g., 'J', 'K', 'T').
        """
        self.thrmcpl_chan = device.name + "/" + device_params["thrmcpl_chan"]
        self.thrmcpl_type = getattr(nidaqmx.constants.ThermocoupleType, thrmcpl_type)
        if device_params["cjc_source"]:
            self.cjc_source = getattr(
                nidaqmx.constants.CJCSource, device_params["cjc_source"]
            )
        else:
            self.cjc_source = ""

        self.nsamples = device_params["nsamples"]

        if not self.nsamples:
            logger.warning("device is not found or not available.")
            return

    def get_tempC(self):
        """Read temperature in Celsius from the thermocouple."""
        with nidaqmx.Task() as task:
            if self.cjc_source:
                task.ai_channels.add_ai_thrmcpl_chan(
                    self.thrmcpl_chan,
                    thermocouple_type=self.thrmcpl_type,
                    cjc_source=self.cjc_source,
                )
            else:
                task.ai_channels.add_ai_thrmcpl_chan(
                    self.thrmcpl_chan,
                    thermocouple_type=self.thrmcpl_type,
                )
            data = task.read(number_of_samples_per_channel=self.nsamples)
            return np.mean(data)
