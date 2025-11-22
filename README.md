# dirigo-ni-digitizer

`dirigo-ni-digitizer` provides a National Instruments (NI-DAQmx)–based implementation of the `Digitizer` interface from [Dirigo](https://dirigo.readthedocs.io/). It allows NI X-series and S-series DAQ boards to be used as digitizers within Dirigo acquisition workflows (e.g. galvo–galvo scanning, analog and photon-counting).

> **Note**  
> This is a hardware plugin for Dirigo and is not intended to be used as a standalone library. 

![PyPI](https://img.shields.io/pypi/v/dirigo-ni-digitizer)


---

## Installation
First install the NI-DAQmx drivers from official NI channels. Then, inside your Python environment (e.g. a conda environment), run:

```bash
pip install dirigo-ni-digitizer
```

It is recommended to verify that your NI device is recognized in NI MAX or the Hardware Configuration Utility before using this plugin.


## Legal Disclaimer
This library is provided "as is" without any warranties, express or implied, including but not limited to the implied warranties of merchantability, fitness for a particular purpose, or non-infringement. The authors are not responsible for any damage to hardware, data loss, or other issues arising from the use or misuse of this library. Users are advised to thoroughly test this library with their specific hardware and configurations before deployment.

This library depends on the NI-DAQmx API and its associated drivers, which must be installed and configured separately. This library interacts with the NI-DAQmx API through the officially supported [nidaqmx Python wrappers](https://nidaqmx-python.readthedocs.io/). Compatibility and performance depend on the proper installation and operation of these third-party components.

This library is an independent implementation based on publicly available documentation from National Instruments. It is not affiliated with, endorsed by, or officially supported by National Instruments.

Use this library at your own risk. Proper operation of hardware and compliance with applicable laws and regulations is the sole responsibility of the user.

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.