__all__ = ["USDPlantExporter"]


def __getattr__(name: str):
    if name == "USDPlantExporter":
        from tomato_recon.export.usd_exporter import USDPlantExporter

        return USDPlantExporter
    raise AttributeError(name)
