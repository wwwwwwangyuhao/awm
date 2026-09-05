"""DSSAT COX, irrigation and fertilizer file handling."""
from __future__ import annotations

import os
from pathlib import Path

from .runner import log_message


class DSSATFileHandler:
    """Write the worker-local COX and management files used by DSSAT."""

    def __init__(
        self,
        exp_year=23,
        action_type="discrete",
        COX_name=None,
        field_name=None,
        weather_name=None,
        plant_date=None,
        emergence_date=None,
        is_phosphorus=0,
        is_potassium=0,
        irrigation_file=None,
        fertilizer_file=None,
        output_path=None,
        verbose=0,
    ) -> None:
        self.output_path = output_path
        self.irrigation_file = irrigation_file
        self.fertilizer_file = fertilizer_file
        self.plant_date = plant_date
        self.emergence_date = emergence_date
        self.is_phosphorus = is_phosphorus
        self.is_potassium = is_potassium
        self.field_name = field_name
        self.weather_name = weather_name
        self.COX_name = COX_name
        self.exp_year = exp_year
        self.num_format = "5d" if action_type == "discrete" else "5.2f"
        self.verbose = int(verbose or 0)

    def write_COX(self) -> None:
        """Write the current XJHX cotton experiment file."""
        try:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            # Li2019 -> DSSAT 4.8.5 mulch mapping used by the current experiment:
            # ALBEDOMULCH=0.12 -> PMALB=0.12; COUVERMULCH=0.75 with
            # PLRS=30 cm -> PMWD=30*0.75=22.5 cm.
            mulch_albedo = float(getattr(self, "mulch_albedo", 0.12))
            mulch_width = float(getattr(self, "mulch_width", 22.5))

            with open(self.output_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"*EXP.DETAILS: JIANGDU PROTOTRON 20{self.exp_year}\n")
                f.write("\n*GENERAL\n")
                f.write("@PEOPLE\nWANG,yuhao\n")
                f.write("@ADDRESS\nZONGCUN,JIANGSU,CHINA\n")
                f.write("@SITE\nJIANGDU,JS,CHA  32.585;119.7;\n")
                f.write(f"@NOTE\n{self.COX_name}    \n")

                f.write("\n*TREATMENTS                        -------------FACTOR LEVELS------------\n")
                f.write("@N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM\n")
                f.write(" 1 1 1 0 V1S1T1                     1  1  0  1  1  1  1  0  0  0  1  0  1\n")

                f.write("\n*CULTIVARS\n")
                f.write("@C CR INGENO CNAME\n")
                f.write(" 1 CO IB0007 IRRIGATION_ONLY\n")

                f.write("\n*FIELDS\n")
                f.write("@L ID_FIELD WSTA....  FLSA  FLOB  FLDT  FLDD  FLDS  FLST SLTX  SLDP  ID_SOIL    FLNAME\n")
                f.write(
                    f" 1 {self.field_name} {self.weather_name}   -99   -99 DR000     0     0     0 "
                    f"CL     -99  {self.field_name}   -99         \n"
                )
                f.write("@L ...........XCRD ...........YCRD .....ELEV .............AREA .SLEN .FLWR .SLAS FLHST FHDUR\n")
                f.write(" 1           0.000           0.000         0                 0     0     0     0   -99   -99\n")
                f.write("@L PMALB  PMWD\n")
                f.write(f" 1 {mulch_albedo:5.2f} {mulch_width:5.1f}\n")

                f.write("\n*INITIAL CONDITIONS\n")
                f.write("@C   PCR ICDAT  ICRT  ICND  ICRN  ICRE  ICWD ICRES ICREN ICREP ICRIP ICRID ICNAME\n")
                f.write(f" 1    CO {self.exp_year}103     0   -99   -99   -99   -99     0     0   -99   -99   -99 -99         \n")
                f.write("@C  ICBL  SH2O  SNH4  SNO3\n")
                f.write(" 1     5  0.37    16     8\n")
                f.write(" 1    10  0.36    16     8\n")
                f.write(" 1    20  0.35     8     4\n")
                f.write(" 1    30  0.38     4     2\n")
                f.write(" 1    50  0.38   0.8   0.4\n")
                f.write(" 1    70  0.38  0.16  0.08\n")
                f.write(" 1   100  0.38  0.03 0.016\n")
                f.write(" 1   130  0.38 0.006 0.001\n")
                f.write(" 1   160  0.38 0.001 0.001\n")
                f.write(" 1   190  0.38     0     0\n")

                f.write("\n*PLANTING DETAILS\n")
                f.write("@P PDATE EDATE  PPOP  PPOE  PLME  PLDS  PLRS  PLRD  PLDP  PLWT  PAGE  PENV  PLPH  SPRL                        PLNAME\n")
                f.write(
                    f" 1 {self.plant_date} {self.emergence_date}    27    24     S     R    30     0     5 "
                    f"  -99   -99   -99   -99   -99                           -99\n"
                )

                f.write("\n*IRRIGATION AND WATER MANAGEMENT\n")
                f.write("@I  EFIR  IDEP  ITHR  IEPT  IOFF  IAME  IAMT IRNAME\n")
                f.write(" 1   -99   -99   -99   -99   -99   -99   -99 -99         \n")
                f.write("@I IDATE  IROP IRVAL\n")
                csv_irrigation = self.read_irrigation() if self.irrigation_file else {}
                for idate, irval in csv_irrigation.items():
                    f.write(f" 1 {idate} IR005 {irval:{self.num_format}}\n")

                f.write("\n*FERTILIZERS (INORGANIC)\n")
                f.write("@F FDATE  FMCD  FACD  FDEP  FAMN  FAMP  FAMK  FAMC  FAMO  FOCD FERNAME\n")
                csv_fertilizer = self.read_fertilizer() if self.fertilizer_file else {}
                if csv_fertilizer:
                    for idate, fert in csv_fertilizer.items():
                        f.write(
                            f" 1 {idate} {fert['FMCD']} {fert['FACD']} "
                            f"{int(fert['FDEP']):5d} "
                            f"{float(fert['FAMN']):{self.num_format}} "
                            f"{fert['FAMP']:5d} {fert['FAMK']:5d} "
                            f"{int(fert['FAMC']):5d} {int(fert['FAMO']):5d} "
                            f"{int(fert['FOCD']):5d} {-99:7d}\n"
                        )
                else:
                    f.write(
                        f" 1 {self.plant_date} FE010 AP005 "
                        f"{2:5d} {0.0:{self.num_format}} {0:5d} {0:5d} "
                        f"{-99:5d} {-99:5d} {-99:5d} {-99:7d}\n"
                    )

                f.write("\n*ENVIRONMENT MODIFICATIONS\n")
                f.write("@E ODATE EDAY  ERAD  EMAX  EMIN  ERAIN ECO2  EDEW  EWIND ENVNAME\n")
                f.write(f" 1 {self.exp_year}080 A   0 A   0 A   0 A   0 A   0 R 376 A   0 A   0 376ppm CO2/35oC\n")

                f.write("\n*SIMULATION CONTROLS\n")
                f.write("@N GENERAL     NYERS NREPS START SDATE RSEED SNAME.................... SMODEL\n")
                f.write(f" 1 GE              1     1     S {self.exp_year}080  2150 JIANGDU PROTOTRON 20{self.exp_year}    CRGRO \n")
                f.write("@N OPTIONS     WATER NITRO SYMBI PHOSP POTAS DISES  CHEM  TILL   CO2\n")
                phosp_val = "Y" if self.is_phosphorus == 1 else "N"
                potas_val = "Y" if self.is_potassium == 1 else "N"
                f.write(f" 1 OP              Y     Y     N     {phosp_val}     {potas_val}     N     N     Y     M\n")
                f.write("@N METHODS     WTHER INCON LIGHT EVAPO INFIL PHOTO HYDRO NSWIT MESOM MESEV MESOL\n")
                f.write(" 1 ME              M     M     E     R     S     L     R     1     G     R     2\n")
                f.write("@N MANAGEMENT  PLANT IRRIG FERTI RESID HARVS\n")
                f.write(" 1 M               R     R     R     R     M\n")
                f.write("@N OUTPUTS     FNAME OVVEW SUMRY FROPT GROUT CAOUT WAOUT NIOUT MIOUT DIOUT VBOSE CHOUT OPOUT\n")
                f.write(" 1 OU              N     Y     Y     1     Y     N     Y     Y     1     N     Y     N     Y\n")

                f.write("\n@  AUTOMATIC MANAGEMENT\n")
                f.write("@N PLANTING    PFRST PLAST PH2OL PH2OU PH2OD PSTMX PSTMN\n")
                f.write(f" 1 PL          {self.exp_year}100 {self.exp_year}114    40   100    30    40    10\n")
                f.write("@N IRRIGATION  IMDEP ITHRL ITHRU IROFF IMETH IRAMT IREFF\n")
                f.write(" 1 IR             30    50   100 GS000 IR001    10     1\n")
                f.write("@N NITROGEN    NMDEP NMTHR NAMNT NCODE NAOFF\n")
                f.write(" 1 NI             30    50    25 FE001 GS000\n")
                f.write("@N RESIDUES    RIPCN RTIME RIDEP\n")
                f.write(" 1 RE            100     1    20\n")
                f.write("@N HARVEST     HFRST HLAST HPCNP HPCNR\n")
                f.write(f" 1 HA              0 {self.exp_year}288   100     0\n")

            log_message(self.verbose, f"DSSAT COX written: {self.output_path}")
        except Exception as exc:
            log_message(self.verbose, f"Failed to write COX: {exc}")
            raise

    def read_irrigation(self) -> dict:
        irrigation_data = {}
        try:
            with open(self.irrigation_file, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        irrigation_data[parts[1]] = float(parts[3])
        except (FileNotFoundError, StopIteration):
            return {}
        return irrigation_data

    def read_fertilizer(self) -> dict:
        fertilizer_data = {}
        try:
            with open(self.fertilizer_file, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 11:
                        fertilizer_data[parts[1]] = {
                            "FMCD": parts[2],
                            "FACD": parts[3],
                            "FDEP": int(float(parts[4])),
                            "FAMN": float(parts[5]),
                            "FAMP": int(float(parts[6])),
                            "FAMK": int(float(parts[7])),
                            "FAMC": int(float(parts[8])),
                            "FAMO": int(float(parts[9])),
                            "FOCD": int(float(parts[10])),
                        }
        except (FileNotFoundError, StopIteration):
            return {}
        return fertilizer_data

    def write_irrigation(self, date_str, water) -> None:
        date_found = False
        idate = str(date_str)
        lines = []
        if os.path.exists(self.irrigation_file):
            with open(self.irrigation_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            data_start_idx = 1 if lines and lines[0].startswith("@I IDATE") else 0
            for i in range(data_start_idx, len(lines)):
                parts = lines[i].split()
                if len(parts) >= 2 and parts[1] == idate:
                    lines[i] = f"1 {idate} IR005 {water:{self.num_format}}\n"
                    date_found = True
                    break
            if not date_found:
                lines.append(f"1 {idate} IR005 {water:{self.num_format}}\n")
        else:
            Path(self.irrigation_file).parent.mkdir(parents=True, exist_ok=True)
            lines = [
                "@I IDATE  IROP IRVAL\n",
                f"1 {idate} IR005 {water:{self.num_format}}\n",
            ]
        with open(self.irrigation_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def write_fertilizer(
        self,
        date_str,
        n_content,
        fdep=2,
        famp=-99,
        famk=-99,
    ) -> None:
        date_found = False
        fdate = str(date_str)
        lines = []
        if os.path.exists(self.fertilizer_file):
            with open(self.fertilizer_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            data_start_idx = 1 if lines and lines[0].startswith("@F FDATE") else 0
            for i in range(data_start_idx, len(lines)):
                parts = lines[i].split()
                if len(parts) >= 2 and parts[1] == fdate:
                    lines[i] = (
                        f"1 {fdate} FE010 AP005 {fdep:5d} "
                        f"{n_content:{self.num_format}} {famp:5d} {famk:5d} "
                        f"{-99:5d} {-99:5d} {-99:5d} {-99:5d}\n"
                    )
                    date_found = True
                    break
            if not date_found:
                lines.append(
                    f"1 {fdate} FE010 AP005 {fdep:5d} "
                    f"{n_content:{self.num_format}} {famp:5d} {famk:5d} "
                    f"{-99:5d} {-99:5d} {-99:5d} {-99:5d}\n"
                )
        else:
            Path(self.fertilizer_file).parent.mkdir(parents=True, exist_ok=True)
            lines = [
                "@F FDATE  FMCD  FACD  FDEP  FAMN  FAMP  FAMK  FAMC  FAMO  FOCD FERNAME\n",
                f"1 {fdate} FE010 AP005 {fdep:5d} "
                f"{n_content:{self.num_format}} {famp:5d} {famk:5d} "
                f"{-99:5d} {-99:5d} {-99:5d} {-99:5d}\n",
            ]
        with open(self.fertilizer_file, "w", encoding="utf-8") as f:
            f.writelines(lines)


__all__ = ["DSSATFileHandler"]
