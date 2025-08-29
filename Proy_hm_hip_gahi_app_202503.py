# Databricks notebook source
# MAGIC %md
# MAGIC ## Ficha de Seguimiento
# MAGIC  
# MAGIC #### Proyecto: SEGUIMIENTO APPLICANT GARANTÍA HIPOTECARIA
# MAGIC **Tablas Fuentes:** catalog_lhcl_prod_bcp.bcp_edv_fabseg.T45988_HM_ADM_HIP_GAHI_202502
# MAGIC  
# MAGIC #### Objetivo:
# MAGIC Realizar el seguimiento del modelo Applicant Garantía Hipotecaria a nivel de clientes Banca Personas
# MAGIC  
# MAGIC #### Tipo:
# MAGIC Python / Pyspark
# MAGIC  
# MAGIC #### Periodicidad:
# MAGIC Mensual
# MAGIC  
# MAGIC #### Versiones:
# MAGIC  
# MAGIC | Versión | Analista   | Responsable Técnico | Fecha       | Descripción            |
# MAGIC |---------|------------|---------------------|-------------|------------------------|
# MAGIC | 1       | Lesly Maza | Gerardo Soto        | 26/11/2024  | Creación del proceso   |
# MAGIC | 2       | Lesly Maza  | Gerardo Soto        | 13/12/2024  | Inclusión de variables del calibrado  |
# MAGIC | 3      | Lesly Maza  | Gerardo Soto        | 26/02/2025  | Uso de índices notebook y seguimiento de discriminación del troncal  |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Seguimiento Applicant Garantia Hipotecaria

# COMMAND ----------

# Importando módulo de seguimiento
import sys
import importlib
sys.path.append('/Workspace/Users/luisfdiaz@bcp.com.pe/Modulo de Seguimiento/')

import SegScore as rmm
importlib.reload(rmm)

# Librerias y fuciones generales
from pyspark.sql.functions import date_format, expr, to_date, date_sub, add_months, col, when, coalesce, trim, broadcast, avg, max, min, lit, concat, window, round as colround, upper, abs as sparkabs
from pyspark.sql import functions as F
from pyspark import StorageLevel
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ####Universo

# COMMAND ----------

from pyspark.sql import SparkSession
base_seguimiento = spark.sql("select distinct * from catalog_lhcl_prod_bcp.bcp_edv_fabseg.T45988_HM_ADM_HIP_GAHI_202503_F")
base_seguimiento = base_seguimiento.withColumn("CODMES", col("CODMES").cast("integer"))
base_seguimiento = base_seguimiento.withColumn("DEF5", when(col("DISTANCIA_DEF24") <= 5, lit(1)).otherwise(lit(0))) 


# COMMAND ----------

base_seguimiento = base_seguimiento.withColumn(
    "INGRESO_CONY_TIT",
    coalesce(col("INGRESO_SOL_CONY"), expr("0")) + coalesce(col("INGRESO_RBM"), expr("0"))
)

# COMMAND ----------

num_filas = base_seguimiento.count()
num_columnas = len(base_seguimiento.columns)

print(f"Número de filas: {num_filas}")
print(f"Número de columnas: {num_columnas}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 0. Definicion de Seguimiento

# COMMAND ----------

pd_vig='PD_GAHI' 
pd_cal='PD_GAHI'
pd_rbm='PD_RBM'
pd_trc='XB_GAHI_PRELIMINAR'
tipo_banda = 'Jeffrey'#Jeffrey Vasiseck
codmes_default = 202302
columna_monto='MTOAPROBADO_SOLES'
titulo='Calibración Applicant App Garantia Hipotecaria'

# COMMAND ----------

condiciones= "CODMES >= 201901 AND DEF_120_BLOQ_REF_0 = 0"

# COMMAND ----------

# Objeto de Seguimiento
Obj_Seg = rmm.MonitorScore_v01(
  nombre_modelo         = 'Personas Modelo Applicant Garantia Hipotecaria',
  codigo_modelo         = 'MOD-BCP-20658',
  tier                  = 'II',
  detalle_seguimiento   = 'Seguimiento Applicant Garantia Hipotecaria',
  mes_seguimiento       = '202503',
  base_monitoreo        = base_seguimiento,
  pd1                   = pd_cal,
  pd2                   = pd_vig,
  pd3                   = pd_rbm,
  monto_credito         = columna_monto,
  query_univ            = "DEF_120_BLOQ_REF_0=0 AND PD_GAHI IS NOT NULL",
  bandas                = tipo_banda,
  codmes_default        = codmes_default,
  meses_ventana_target  = 24,
  meses_atraso_target   = 4
)

# COMMAND ----------

from pyspark.sql.functions import col, substring

# COMMAND ----------

Obj_Seg.base_monitoreo = Obj_Seg.base_monitoreo.withColumn("PD_GAHI_PRELIMINAR", expr("1 / (1 + EXP(-XB_GAHI_PRELIMINAR))"))

# COMMAND ----------

Obj_Seg.base_monitoreo = Obj_Seg.base_monitoreo.withColumn("mes", col("codmes").cast("string"))

# COMMAND ----------

Obj_Seg.base_monitoreo = Obj_Seg.base_monitoreo.withColumn("codyear", substring(col("mes"), 1, 4))

# COMMAND ----------

Obj_Seg.base_monitoreo.select("codyear").show(10)

# COMMAND ----------

 Obj_Seg.base_monitoreo = Obj_Seg.base_monitoreo.withColumn(
    "RAN_PLAZO",
    F.when(F.col("RAN_PLAZO") == "HASTA 10", "1  HASTA 10")
     .when(F.col("RAN_PLAZO") == "<10 - 15]", "2 <10 - 15]")
     .when(F.col("RAN_PLAZO") == "<15 - 20]", "3 <15 - 20]")
     .when(F.col("RAN_PLAZO") == "MAS DE 20", "4 MAS DE 20")
     .otherwise(F.col("RAN_PLAZO"))
)
Obj_Seg.base_monitoreo.select("RAN_PLAZO").show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Vista General

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.1. Calibracion Cuentas y Montos

# COMMAND ----------

# import inspect
# inspect.signature(Obj_Seg.EvolCal).parameters

# COMMAND ----------

# DBTITLE 1,missing pd vigente
result = base_seguimiento.where(col("CODMES")>=202101).groupBy("CODMES").agg(
    F.sum(F.when(F.col("PD_GAHI").isNull(), 1).otherwise(0)).alias("MISS_PD"),
    F.sum(F.when(F.col("PD_GAHI").isNull() & F.col("CODSOLICITUD").isNull(), 1).otherwise(0)).alias("MISS_SOL"),
    F.count("*").alias("N")
).orderBy("CODMES")

result.show(200)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Total

# COMMAND ----------

Obj_Seg_cal, rc, rm, fc, fm = Obj_Seg.EvolCal(
  # Configuración de filtro de uso y pd de calibración
  query_uso          = "CODMES>=202101",
  pd_seg             = pd_cal,
  pd_comparacion     = True,
  proys_def_temp     = 6,
  proys_mor_temp     = 0,
  rd_aux             = 'RD21',
  rd_aux2            = 'RD18', # Corregido
  mora_temp          = 'MORA_30_3',

  # Configuración de proyecciones por cuentas
  proy_def_temp_xc   = "np.where( (calxcuentas['CODMES'] == 202302) | ( calxcuentas['CODMES'] == 202303)| ( calxcuentas['CODMES'] == 202304), (met1_xc) +0.005, (met1_xc) )", #All_proy met1_xc met2_xc met3_xc
  suav_def_temp_xc   = 1,
  proy_mor_temp_xc   = 'det1_xc',
  suav_mor_temp_xc   = 1,
  prof_hist_xc       = 14,

  # Configuración de proyecciones por montos
  proy_def_temp_xm   = "np.where( (calxmontos['CODMES'] == 202302) | ( calxmontos['CODMES'] == 202303)| ( calxmontos['CODMES'] == 202304) | ( calxmontos['CODMES'] == 202306) | ( calxmontos['CODMES'] == 202307), (met1_xm) +0.005, (met1_xm) )", #All_proy met1_xm met2_xm met3_xm
  suav_def_temp_xm   = 1,
  proy_mor_temp_xm   = 'det1_xm',
  suav_mor_temp_xm   = 1,
  prof_hist_xm       = 14,
  #fact_to_proy_xc    = [fc_af, fc_con],
  #fact_to_proy_xm    = [fm_af, fm_con],

  # Detalles del gráfico de calibración
  titulo             = titulo,
  pos_leyenda        = (0.5, -0.25),
  eje_y_xc           = 0.15,
  eje_y_xm           = 0.15,
  dim_grafico        = (25, 6.5),
  punt_mora          = 85, #Tamaño de puntos de mora temprana
  etiquetas          =True,
  pos_etiquetas_xm   =10,
  pos_etiquetas_xc   =10,
  tamaño_etiqueta    =20,

  # Exportar factores para vista driver
  exportar_factores  = True
)

#met 1: factor percentilico
#met 2: percentelica promediado con perc 50
#met 3: tendencial por var RD

# COMMAND ----------

fc_1

# COMMAND ----------

rc

# COMMAND ----------

rm

# COMMAND ----------

periodo_proy_1=202303
periodo_proy_6=202308

# COMMAND ----------

#PD y RD proy
rm['PD_GAHI'][(rm['CODMES']>=periodo_proy_1) & (rm['CODMES']<=periodo_proy_6)].mean(), rm['RD_PROY_DEF'][(rm['CODMES']>=periodo_proy_1) & (rm['CODMES']<=periodo_proy_6)].mean()

# COMMAND ----------

rc['PD_GAHI'][(rc['CODMES']>=periodo_proy_1) & (rc['CODMES']<=periodo_proy_6)].mean(), rc['RD_PROY_DEF'][(rc['CODMES']>=periodo_proy_1) & (rc['CODMES']<=periodo_proy_6)].mean()

# COMMAND ----------

# Variables a categoricas
columnas = ["PD_GAHI"]
condicion_filtro = "CODMES BETWEEN 202101 AND 202410 AND DEF_120_BLOQ_REF_0 = 0"
Obj_Seg.base_monitoreo = categorize_by_quintiles(Obj_Seg.base_monitoreo, columnas, condicion_filtro)

# COMMAND ----------

# Variables a categoricas
columnas = ["XB_GAHI_PRELIMINAR"]
condicion_filtro = "CODMES BETWEEN 202101 AND 202410 AND DEF_120_BLOQ_REF_0 = 0"
Obj_Seg.base_monitoreo = categorize_by_quintiles(Obj_Seg.base_monitoreo, columnas, condicion_filtro)

# COMMAND ----------

driver = 'RIESGO_ZONA2'
Obj_SegPSI, mix_n, mix_m, psi_c, psi_m = Obj_Seg.MixPSI(
  # Selección del driver y filtro de uso
  driver             = driver,
  query_uso          = 'CODMES >= 201701',
  cast_int           = False,

  # Ventana de construcción o más antigua
  codmes_inicio1     = 202101,
  codmes_fin1        = 202212,

  # Periodo reciente
  codmes_inicio2     = 202301,
  codmes_fin2        = 202410,

  # Detalles del gráfico
  titulo             = driver,
  dim_grafico        = (22.5, 6),
  pos_leyenda        = (0.5, -0.25)
)

# COMMAND ----------

Obj_SegDis3, evoldis, ks= Obj_Seg.EvolDis(
  query_filtro = "CODMES>=201701 AND DEF_120_BLOQ_REF_0 = 0",
  pd_dis       = pd_trc,
  codmes_ini   = 201701,
  codmes_fin   = 202312,
  amplitud     = 24,
  disc_target= [24, 21,18],
  umbral_sup   = 0.4,
  umbral_inf   = 0.3,
  moviles      = True,
  etiquetas= True,
  tamaño_etiqueta=20,
  dim_grafico  = (35, 7),
  pos_leyenda  = (0.5, -0.3)
)

# COMMAND ----------

driver = 'PD_GAHI_Q'
Obj_SegPSI, mix_n, mix_m, psi_c, psi_m = Obj_Seg.MixPSI(
  # Selección del driver y filtro de uso
  driver             = driver,
  query_uso          = 'CODMES >= 202101',
  cast_int           = False,

  # Ventana de construcción o más antigua
  codmes_inicio1     = 202101,
  codmes_fin1        = 202212,

  # Periodo reciente
  codmes_inicio2     = 202301,
  codmes_fin2        = 202410,

  # Detalles del gráfico
  titulo             = driver,
  dim_grafico        = (22.5, 6),
  pos_leyenda        = (0.5, -0.25)
)

# COMMAND ----------

driver = 'PD_GAHI_Q'
Obj_SegPSI, mix_n, mix_m, psi_c, psi_m = Obj_Seg.EvolPSI(
  # Selección del driver y filtro de uso
  driver             = driver,
  query_uso          = 'CODMES >= 202101',
  cast_int           = False,

  # Ventana de construcción o más antigua
  codmes_inicio1     = 202101,
  codmes_fin1        = 202212,

  # Periodo reciente
  codmes_inicio2     = 202301,
  codmes_fin2        = 202501,

  # Detalles del gráfico
  titulo             = driver,
  dim_grafico        = (22.5, 6),
  pos_leyenda        = (0.5, -0.25),
  alpha              = 1,
  color              = ['','']
)

# COMMAND ----------

Obj_Seg.base_monitoreo.columns

# COMMAND ----------

Obj_Seg_dri, rc_dr, rm_dri = Obj_Seg.CalDri(
  driver             = 'PD_GAHI_Q',
  query_uso          = f"(CODMES BETWEEN 202012 AND 202302) AND CODMES NOT IN (202201,202208) AND PD_GAHI_Q NOT LIKE '%Missing%' ",
  pd_seg             = pd_cal,
  rd_aux             = 'RD6',
  pd_comparacion     = True,
  factores_cuentas   = fc,
  factores_montos    = fm,
  mat_importancia    = 0,
  cast_int           = True, 
  titulo             = f'Rango de PD',
  dim_grafico        = (25, 6),
  etiquetas          = True,
  pos_etiquetas_xc   = [10, 10, 10, 10, 10, 10, 10, 10],
  pos_etiquetas_xm   = [10, 10, 10, 10, 10, 10, 10, 10],
  pos_leyenda        = (0.5, -0.2),
  punt_mora          = 100,
  iv                 = False
) 

# COMMAND ----------

rm_dri

# COMMAND ----------

Obj_Seg_dri, rc_dr, rm_dri = Obj_Seg.CalDri(
  driver             = 'codyear',
  query_uso          = f"(CODMES BETWEEN 202012 AND 202302) AND CODMES NOT IN (202201,202208) AND codyear NOT LIKE  '%Missing%' ",
  pd_seg             = pd_cal,
  rd_aux             = 'RD6',
  pd_comparacion     = True,
  factores_cuentas   = fc,
  factores_montos    = fm,
  mat_importancia    = 0,
  cast_int           = True, 
  titulo             = f'Vista Anual',
  dim_grafico        = (25, 6),
  etiquetas          = True,
  pos_etiquetas_xc   = [10, 10, 10, 10, 10, 10, 10, 10],
  pos_etiquetas_xm   = [10, 10, 10, 10, 10, 10, 10, 10],
  pos_leyenda        = (0.5, -0.2),
  punt_mora          = 100,
  iv                 = False
) 

# COMMAND ----------

rc_dr

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.2. Discriminación

# COMMAND ----------

import inspect
inspect.signature(Obj_Seg.EvolDis).parameters

# COMMAND ----------

Obj_SegDis, evoldis_gini, evoldis_ks = Obj_Seg.EvolDis(
  query_filtro = "CODMES>=201901 AND DEF_120_BLOQ_REF_0 = 0",
  pd_dis       = pd_cal,
  codmes_ini   = 202201,
  codmes_fin   = 202312,
  amplitud     = 24,
  disc_target= [24, 21,18],
  umbral_sup   = 0.5,
  umbral_inf   = 0.4,
  moviles      = True,
  etiquetas= True,
  tamaño_etiqueta=20,
  dim_grafico  = (35, 7),
  pos_leyenda  = (0.5, -0.3)
)

# COMMAND ----------

evoldis_gini['GINI24'].mean()

# COMMAND ----------

Obj_SegDis3, evoldis_trc,evolks_trc = Obj_Seg.EvolDis(
  query_filtro = "CODMES>=201901 AND DEF_120_BLOQ_REF_0 = 0",
  pd_dis       = pd_trc,
  codmes_ini   = 202110,
  codmes_fin   = 202312,
  amplitud     = 24,
  disc_target= [24, 21,18],
  umbral_sup   = 0.4,
  umbral_inf   = 0.3,
  moviles      = True,
  etiquetas= True,
  tamaño_etiqueta=20,
  dim_grafico  = (35, 7),
  pos_leyenda  = (0.5, -0.3)
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Vista Driver

# COMMAND ----------

#Cuartiles
from pyspark.sql import functions as F

def categorize_by_quintiles(df, column_names, filter_condition, quintile_probabilities=[0.25, 0.5, 0.75, 1.0], relative_error=0.01):
    for column in column_names:
        # Filtrar y calcular los quintiles
        quintiles = df.filter(filter_condition).approxQuantile(column, quintile_probabilities, relative_error)
        Minimo = df.agg({column: "min"}).collect()[0][0]
        
        # Crear la nueva columna categorizada
        df = df.withColumn(
            f'{column}_Q',
            F.when(F.col(column) <= quintiles[0], f"1. {format(Minimo, '.3f')} - {format(quintiles[0], '.3f')}")
             .when(F.col(column) <= quintiles[1], f"2. {format(quintiles[0], '.3f')} - {format(quintiles[1], '.3f')}")
             .when(F.col(column) <= quintiles[2], f"3. {format(quintiles[1], '.3f')} - {format(quintiles[2], '.3f')}")
             .when(F.col(column) <= quintiles[3], f"4. {format(quintiles[2], '.3f')} - {format(quintiles[3], '.3f')}")
             .otherwise("98. Missing")
        )
    return df

# COMMAND ----------

# Definir las listas de variables
var_neg = ["MONTO_R", "RAN_PLAZO", "PRS_TASA_Q", "SEGMENTO_BANCA_F", "TIPO_EMPLEO_F", "EDAD_R"]
var_neg_st = ["MTOAPROBADO_SOLES", "CTDPLAZOAPROBADO", "PRS_TASA", "SEGMENTO_BANCA_F", "TIPO_EMPLEO_F", "EDAD"]
var_gen = ["PAUTA", "LTV_CORREGIDO_0_Q", "INGRESO_SOL_CONY_Q", "END_TOT_SF_CY_Q", "FLG_PDH_2_3", "INGRESO_RBM_Q", "END_TOT_SF_Q", "MARCA_HML_T", "INGRESO_CONY_TIT_Q"]
var_gen_st=["PAUTA", "LTV_CORREGIDO_0", "INGRESO_SOL_CONY", "END_TOT_SF_CY", "FLG_PDH_2_3", "INGRESO_RBM", "END_TOT_SF", "MARCA_HML_T", "INGRESO_CONY_TIT"]

var_cal = ["PD_GAHI_Q", "PD_GAHI_PRELIMINAR_Q", "FLG_MONTO_CAL","PD_GAHI_Q", "PD_GAHI_PRELIMINAR_Q", "FLG_MONTO_CAL"]
var_cal_st = ["PD_GAHI", "PD_GAHI_PRELIMINAR", "FLG_MONTO_CAL","PD_GAHI_Q", "PD_GAHI_PRELIMINAR_Q", "FLG_MONTO_CAL"]


xb_mod = ["XB_NUE_APP_CEF", "XB_MODUL_GAHI"]
var_mod = [ "RIESGO_ZONA2","CTDPLAZOAPROBADO_TP_Q","LTV_NEW_CAL_Q","NFT_RIESGO_ZONA2","LTV_NEW_CAL_Q", "INGRESO_SOL_CONY_TP_Q"]
var_mod_st = ["RIESGO_ZONA2_ST","CTDPLAZOAPROBADO_A2","LTV_NEW_A2","LTV_NEW_A2", "INGRESO_SOL_CONY_D3"]

# COMMAND ----------

Obj_Seg.base_monitoreo.select("FLG_MONTO_CAL").show(10)

# COMMAND ----------

# Variables a categoricas
columnas = ["PD_GAHI_PRELIMINAR"]
condicion_filtro = "CODMES BETWEEN 202101 AND 202412 AND DEF_120_BLOQ_REF_0 = 0"
Obj_Seg.base_monitoreo = categorize_by_quintiles(Obj_Seg.base_monitoreo, columnas, condicion_filtro)

# COMMAND ----------

# Variables a categoricas
columnas = ["INGRESO_SOL_CONY_TP"]
condicion_filtro = "CODMES BETWEEN 202101 AND 202412 AND DEF_120_BLOQ_REF_0 = 0"
Obj_Seg.base_monitoreo = categorize_by_quintiles(Obj_Seg.base_monitoreo, columnas, condicion_filtro)

# COMMAND ----------

# Variables a categoricas
columnas = ["LTV_NEW_CAL"]
condicion_filtro = "CODMES BETWEEN 202101 AND 202412 AND DEF_120_BLOQ_REF_0 = 0"
Obj_Seg.base_monitoreo = categorize_by_quintiles(Obj_Seg.base_monitoreo, columnas, condicion_filtro)

# COMMAND ----------

# Variables a categoricas
columnas = ["CTDPLAZOAPROBADO_TP"]
condicion_filtro = "CODMES BETWEEN 202101 AND 202412 AND DEF_120_BLOQ_REF_0 = 0"
Obj_Seg.base_monitoreo = categorize_by_quintiles(Obj_Seg.base_monitoreo, columnas, condicion_filtro)

# COMMAND ----------

# Variables a categoricas
columnas = ["EDAD"]
condicion_filtro = "CODMES BETWEEN 202101 AND 202412 AND DEF_120_BLOQ_REF_0 = 0"
Obj_Seg.base_monitoreo = categorize_by_quintiles(Obj_Seg.base_monitoreo, columnas, condicion_filtro)

# COMMAND ----------

# Variables a categoricas
columnas = ["END_TOT_SF"]
condicion_filtro = "CODMES BETWEEN 202101 AND 202412 AND DEF_120_BLOQ_REF_0 = 0"
Obj_Seg.base_monitoreo = categorize_by_quintiles(Obj_Seg.base_monitoreo, columnas, condicion_filtro)

# COMMAND ----------

# Variables a categoricas
columnas = ["END_TOT_SF_CY"]
condicion_filtro = "CODMES BETWEEN 202101 AND 202412 AND DEF_120_BLOQ_REF_0 = 0"
Obj_Seg.base_monitoreo = categorize_by_quintiles(Obj_Seg.base_monitoreo, columnas, condicion_filtro)

# COMMAND ----------

# Variables a categoricas
columnas = ["LTV_CORREGIDO_0"]
condicion_filtro = "CODMES BETWEEN 202101 AND 202412 AND DEF_120_BLOQ_REF_0 = 0"
Obj_Seg.base_monitoreo = categorize_by_quintiles(Obj_Seg.base_monitoreo, columnas, condicion_filtro)

# COMMAND ----------

# Variables a categoricas
columnas = ["INGRESO_SOL_CONY"]
condicion_filtro = "CODMES BETWEEN 202101 AND 202412 AND DEF_120_BLOQ_REF_0 = 0"
Obj_Seg.base_monitoreo = categorize_by_quintiles(Obj_Seg.base_monitoreo, columnas, condicion_filtro)

# COMMAND ----------

# Variables a categoricas
columnas = ["INGRESO_RBM"]
condicion_filtro = "CODMES BETWEEN 202101 AND 202412 AND DEF_120_BLOQ_REF_0 = 0"
Obj_Seg.base_monitoreo = categorize_by_quintiles(Obj_Seg.base_monitoreo, columnas, condicion_filtro)

# COMMAND ----------

# Variables a categoricas
columnas = ["PRS_TASA"]
condicion_filtro = "CODMES BETWEEN 202101 AND 202412 AND DEF_120_BLOQ_REF_0 = 0"
Obj_Seg.base_monitoreo = categorize_by_quintiles(Obj_Seg.base_monitoreo, columnas, condicion_filtro)

# COMMAND ----------

# Variables a categoricas
columnas = ["INGRESO_CONY_TIT"]
condicion_filtro = "CODMES BETWEEN 202101 AND 202412 AND DEF_120_BLOQ_REF_0 = 0"
Obj_Seg.base_monitoreo = categorize_by_quintiles(Obj_Seg.base_monitoreo, columnas, condicion_filtro)

# COMMAND ----------

suite_titular = spark.sql("""
    SELECT a.*
    FROM catalog_lhcl_prod_bcp.bcp_ddv_rbmrbmper_reportlogevaluacion_vu.hd_personasolicitudevaluacioncreditohipoticariorbmper a
    INNER JOIN (
        SELECT numsecuencialsolicitudcreditohipotecario, MAX(fecrutina) AS max_fecrutina
        FROM catalog_lhcl_prod_bcp.bcp_ddv_rbmrbmper_reportlogevaluacion_vu.hd_personasolicitudevaluacioncreditohipoticariorbmper 
        WHERE destiporelacionadoevaluacion = 'TITULAR'
        GROUP BY numsecuencialsolicitudcreditohipotecario
    ) b
    ON a.numsecuencialsolicitudcreditohipotecario = b.numsecuencialsolicitudcreditohipotecario AND a.fecrutina = b.max_fecrutina
    WHERE a.destiporelacionadoevaluacion = 'TITULAR'
""")

# COMMAND ----------

Obj_Seg.base_monitoreo = Obj_Seg.base_monitoreo.join(
    suite_titular.selectExpr("numsecuencialsolicitudcreditohipotecario as codsolicitudappriesgo","desdistrito as SUITE_desdistrito","desprovincia as SUITE_desprovincia"),
    on = ["codsolicitudappriesgo"],
    how = 'left'
)

# COMMAND ----------

df=Obj_Seg.base_monitoreo

# COMMAND ----------

Obj_Seg.base_monitoreo  = df.withColumn("NFT_RIESGO_ZONA2", 
                   when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'ANCON-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'BARRANCO-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'CALLAO') & (df.NFT_distritoinmueble == 'BELLAVISTA-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'CALLAO') & (df.NFT_distritoinmueble == 'CARMEN DE LA LEGUA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'LIMA CERCADO-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'CHACLACAYO-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'CHORRILLOS-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'CIENEGUILLA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'COMAS-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'INDEPENDENCIA-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'JESUS MARIA-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'LA MOLINA-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'CALLAO') & (df.NFT_distritoinmueble == 'LA PERLA-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'CALLAO') & (df.NFT_distritoinmueble == 'LA PUNTA-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'LA VICTORIA-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'LINCE-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'LOS OLIVOS-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'LURIGANCHO-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'LURIN-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'MAGDALENA DEL MAR'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'MIRAFLORES-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'PUCUSANA-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'PUEBLO LIBRE-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'PUNTA NEGRA-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'RIMAC-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'SAN BARTOLO-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'SAN BORJA-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'SAN ISIDRO-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'SAN LUIS-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'SAN MARTIN DE PORRES'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'SAN MIGUEL-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'SANTA ANITA-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble.isin(['SANTA MARIA DEL MAR','SANTA MARIA-LIMA'])), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'SANTA ROSA-LIMA'), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble.isin(['SANTIAGO DE SURCO','SURCO-LIMA'])), 'BAJO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'SURQUILLO-LIMA'), 'BAJO')
                   .when(df.NFT_departamentoinmueble == 'ICA', 'ALTO')
                   .when(df.NFT_departamentoinmueble == 'PIURA', 'ALTO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'ATE-LIMA'), 'ALTO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'BREÑA-LIMA'), 'ALTO')
                   .when((df.NFT_departamentoinmueble == 'CALLAO') & (df.NFT_distritoinmueble == 'CALLAO-LIMA'), 'ALTO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'CARABAYLLO-LIMA'), 'ALTO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'EL AGUSTINO-LIMA'), 'ALTO')
                   .when((df.NFT_departamentoinmueble == 'CALLAO') & (df.NFT_distritoinmueble == 'MI PERU-CALLAO'), 'ALTO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'PACHACAMAC-LIMA'), 'ALTO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'PUENTE PIEDRA-LIMA'), 'ALTO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'SAN JUAN LURIGANCHO'), 'ALTO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'SAN JUAN MIRAFLORES'), 'ALTO')
                   .when((df.NFT_departamentoinmueble == 'CALLAO') & (df.NFT_distritoinmueble == 'VENTANILLA-CALLAO'), 'ALTO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'VILLA EL SALVADOR'), 'ALTO')
                   .when((df.NFT_departamentoinmueble == 'LIMA') & (df.NFT_distritoinmueble == 'VILLA MARIA DEL TRIUNFO-LIMA'), 'ALTO')
                   .otherwise('BAJO'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2.1 Calibracion Driver

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 2.1.1. Drivers negocio

# COMMAND ----------

# fc = pd.DataFrame()
# fc['CODMES'] = [ 202301, 202302, 202303, 202304, 202305,202306]
# fc['FAC_CAL_DRI_N'] = [1,1 ,1 ,1 ,1 ,1]
 
# fm = pd.DataFrame()
# fm['CODMES'] =  [ 202301, 202302, 202303, 202304, 202305,202306]
# fm['FAC_CAL_DRI_M'] = [1,1 ,1 ,1 ,1 ,1]

# COMMAND ----------

sorted(base_seguimiento.columns)

# COMMAND ----------


caldri, cubo = Obj_Seg.CalVar(
  # Parametros de figura individual
  query_uso          = f"(CODMES BETWEEN 202012 AND 202302) AND CODMES NOT IN (202201,202208)",
  pd_seg             = 'PD_GAHI',
  rd_aux             = 'RD12',
  pd_comparacion     = True, #Para tomar mas de un PD
  factores_cuentas   = fc,
  factores_montos    = fm,
  mat_importancia    = 5, # minimo % de materialidad por bucket mostrado
  cast_int           = True, #para los flag los cambia a entero
  dim_grafico        = (25, 6),
  pos_leyenda        = (0.5, -0.25),
  punt_mora          = 70, # tamaño max 1000
 
  # Parametros de figura final
  tipo_vista = 'M',
  ncolumnas = 3,
  nfilas    = 2,
  variables = var_neg,
  dim_grafico_total = (25, 10),
  ncol_leyenda     = 5,
  etiquetas_total=True, 
  pos_etiqueta      = 0.05,
  vspacio           = -0.5,
  hspacio           = 1,
  filtro_var="not like '%issing%'"# para no mostrar missing
)

# COMMAND ----------

cubo

# COMMAND ----------

caldri = Obj_Seg.CalVar(
  # Parametros de figura individual
  query_uso          = f"(CODMES BETWEEN 202012 AND 202302) AND CODMES NOT IN (202201,202208)",
  pd_seg             = 'PD_GAHI',
  rd_aux             = 'RD12',
  pd_comparacion     = True, #Para tomar mas de un PD
  factores_cuentas   = fc,
  factores_montos    = fm,
  mat_importancia    = 5, # minimo % de materialidad por bucket mostrado
  cast_int           = True, #para los flag los cambia a entero
  dim_grafico        = (25, 6),
  pos_leyenda        = (0.5, -0.25),
  punt_mora          = 70, # tamaño max 1000
 
  # Parametros de figura final
  tipo_vista = 'C',
  ncolumnas = 3,
  nfilas    = 2,
  variables = var_neg,
  dim_grafico_total = (25, 10),
  ncol_leyenda     = 5,
  etiquetas_total=True, 
  pos_etiqueta      = 0.05,
  vspacio           = -0.5,
  hspacio           = 1,
  filtro_var="not like '%issing%'"# para no mostrar missing
)

# COMMAND ----------

psi_dri = Obj_Seg.MixPSIVar(
  # Parametros de figura individual
    query_uso='CODMES >= 202101',
    cast_int=True,
    codmes_inicio1=202101,
    codmes_fin1=202112,
    codmes_inicio2=202301,
    codmes_fin2=202312,
    dim_grafico=(25, 6),
    pos_leyenda=(0.5, -0.25),
 
    # Parametros de figura final
    tipo_vista = 'C',
    ncolumnas = 3,
    nfilas    = 2,
    variables = var_neg,
    dim_grafico_total = (25, 10),
    ncol_leyenda=5,
    tamaño_leyenda = 8.25,
    vspacio=-0.25,
    hspacio=0
)

# COMMAND ----------

gini = Obj_Seg.EvolDisVar(
  # Parametros de figura individual
    query_filtro = '',
    codmes_ini   = 202110,
    codmes_fin   = 202312,
    amplitud     = 24,
    disc_target  = [24, 21,18],
    umbral_sup   = 0.15,
    umbral_inf   = 0.05,
    #umbral_opc   = 0.1,
    moviles      = True,
    dim_grafico  = (15, 6),
    pos_leyenda  = (0.5, -0.3),
 
    etiquetas    = True,
    tamaño_etiqueta    =12,

    # Parametros de figura final
    ncolumnas = 3,
    nfilas    = 2,
    variables = var_neg_st,
    dim_grafico_total = (25, 10),
    ncol_leyenda = 5,
    vspacio=-0.25,
    hspacio=4
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 2.1.2. Drivers generales

# COMMAND ----------


caldri = Obj_Seg.CalVar(
  # Parametros de figura individual
  query_uso          = f"(CODMES BETWEEN 202012 AND 202302) AND CODMES NOT IN (202201,202208)",
  pd_seg             = 'PD_GAHI',
  rd_aux             = 'RD12',
  pd_comparacion     = True, #Para tomar mas de un PD
  factores_cuentas   = fc,
  factores_montos    = fm,
  mat_importancia    = 5, # minimo % de materialidad por bucket mostrado
  cast_int           = True, #para los flag los cambia a entero
  dim_grafico        = (25, 6),
  pos_leyenda        = (0.5, -0.2),
  punt_mora          = 70, # tamaño max 1000
 
  # Parametros de figura final
  tipo_vista = 'M',
  ncolumnas = 3,
  nfilas    = 3,
  variables = var_gen,
  dim_grafico_total = (25, 10),
  ncol_leyenda     = 5,
  etiquetas_total=True, 
  pos_etiqueta      = 0.05,
  vspacio           = -0.5,
  hspacio           = 1,
  filtro_var="not like '%issing%'"# para no mostrar missing
)

# COMMAND ----------

psi_dri = Obj_Seg.MixPSIVar(
  # Parametros de figura individual
    query_uso='CODMES >= 202101',
    cast_int=True,
    codmes_inicio1=202101,
    codmes_fin1=202112,
    codmes_inicio2=202301,
    codmes_fin2=202312,
    dim_grafico=(22.5, 6),
    pos_leyenda=(0.5, -0.35),
 
    # Parametros de figura final
    tipo_vista = 'M',
    ncolumnas = 3,
    nfilas    = 3,
    variables = var_gen,
    dim_grafico_total = (25, 15),
    ncol_leyenda=8,
    tamaño_leyenda = 8.25,
    vspacio=-0.25,
    hspacio=0
)

# COMMAND ----------

gini = Obj_Seg.EvolDisVar(
  # Parametros de figura individual
    query_filtro = '',
    codmes_ini   = 202201,
    codmes_fin   = 202312,
    amplitud     = 24,
    disc_target  = [24, 21,18],
    umbral_sup   = 0.15,
    umbral_inf   = 0.05,
    #umbral_opc   = 0.1,
    moviles      = True,
    dim_grafico  = (22.5, 6),
    pos_leyenda  = (0.5, -0.35),
 
    etiquetas    = True,
    tamaño_etiqueta    =12,

    # Parametros de figura final
    ncolumnas = 3,
    nfilas    = 3,
    variables = var_gen_st,
    dim_grafico_total = (25, 15),
    ncol_leyenda = 5,
    vspacio=-0.25,
    hspacio=4
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 2.1.3 Drivers Calibrado

# COMMAND ----------

caldri = Obj_Seg.CalVar(
  # Parametros de figura individual
  query_uso          = f"(CODMES BETWEEN 202012 AND 202302) AND CODMES NOT IN (202201,202208)",
  pd_seg             = 'PD_GAHI',
  rd_aux             = 'RD12',
  pd_comparacion     = True, #Para tomar mas de un PD
  factores_cuentas   = fc,
  factores_montos    = fm,
  mat_importancia    = 5, # minimo % de materialidad por bucket mostrado
  cast_int           = True, #para los flag los cambia a entero
  dim_grafico        = (25, 6),
  pos_leyenda        = (0.5, -0.25),
  punt_mora          = 70, # tamaño max 1000
 
  # Parametros de figura final
  tipo_vista = 'M',
  ncolumnas = 3,
  nfilas    = 2,
  variables = var_cal,
  dim_grafico_total = (25, 10),
  ncol_leyenda     = 5,
  etiquetas_total=True, 
  pos_etiqueta      = 0.05,
  vspacio           = -0.5,
  hspacio           = 1,
  filtro_var="not like '%issing%'"# para no mostrar missing
)

# COMMAND ----------

psi_dri = Obj_Seg.MixPSIVar(
  # Parametros de figura individual
    query_uso='CODMES >= 202101',
    cast_int=True,
    codmes_inicio1=202101,
    codmes_fin1=202112,
    codmes_inicio2=202301,
    codmes_fin2=202312,
    dim_grafico=(25, 6),
    pos_leyenda=(0.5, -0.25),
 
    # Parametros de figura final
    tipo_vista = 'M',
    ncolumnas = 3,
    nfilas    = 2,
    variables = var_cal,
    dim_grafico_total = (25, 10),
    ncol_leyenda=5,
    tamaño_leyenda = 8.25,
    vspacio=-0.25,
    hspacio=0
)

# COMMAND ----------

gini = Obj_Seg.EvolDisVar(
  # Parametros de figura individual
    query_filtro = '',
    codmes_ini   = 202201,
    codmes_fin   = 202312,
    amplitud     = 24,
    disc_target  = [24, 21,18],
    umbral_sup   = 0.15,
    umbral_inf   = 0.05,
    #umbral_opc   = 0.1,
    moviles      = True,
    dim_grafico  = (15, 6),
    pos_leyenda  = (0.5, -0.3),
 
    etiquetas    = True,
    tamaño_etiqueta    =12,

    # Parametros de figura final
    ncolumnas = 3,
    nfilas    = 2,
    variables = var_cal_st,
    dim_grafico_total = (25, 10),
    ncol_leyenda = 5,
    vspacio=-0.25,
    hspacio=4
  )

# COMMAND ----------

Obj_Seg_cal, rc_bck, rm_bck, fc, fm = Obj_Seg.EvolCal(
  # Configuración de filtro de uso y pd de calibración
  query_uso          = "CODMES>=202101 and FLG_MONTO_CAL=1",
  pd_seg             = pd_cal,
  pd_comparacion     = True,
  proys_def_temp     = 6,
  proys_mor_temp     = 0,
  rd_aux             = 'RD21',
  rd_aux2            = 'RD12', # Corregido
  mora_temp          = 'MORA_30_3',

  # Configuración de proyecciones por cuentas
  proy_def_temp_xc   = 'All_proy', #All_proy met1_xc met2_xc met3_xc
  suav_def_temp_xc   = 1,
  proy_mor_temp_xc   = 'All_proy', #All_proy det1_xm det2_xm det3_xm
  suav_mor_temp_xc   = 1,
  prof_hist_xc       = 14,

  # Configuración de proyecciones por montos
  proy_def_temp_xm   = "All_proy", #All_proy met1_xm met2_xm met3_xm
  suav_def_temp_xm   = 1,
  proy_mor_temp_xm   = 'All_proy', #All_proy det1_xm det2_xm det3_xm
  suav_mor_temp_xm   = 1,
  prof_hist_xm       = 14,
  #fact_to_proy_xc    = [fc_1, fc_2],
  #fact_to_proy_xm    = [fm_1, fm_2],

  # Detalles del gráfico de calibración
  titulo             = titulo,
  pos_leyenda        = (0.5, -0.25),
  eje_y_xc           = 0.25,
  eje_y_xm           = 0.25,
  dim_grafico        = (25, 6.5),
  punt_mora          = 85, #Tamaño de puntos de mora temprana
  etiquetas          =True,
  pos_etiquetas_xm   =10,
  pos_etiquetas_xc   =10,
  tamaño_etiqueta    =20,

  # Exportar factores para vista driver
  exportar_factores  = True,
  mat_porc=-24
)

# COMMAND ----------

Obj_Seg_cal, rc_bck1, rm_bck1, fc, fm = Obj_Seg.EvolCal(
  # Configuración de filtro de uso y pd de calibración
  query_uso          = "CODMES>=202101 and FLG_MONTO_CAL=0",
  pd_seg             = pd_cal,
  pd_comparacion     = True,
  proys_def_temp     = 6,
  proys_mor_temp     = 0,
  rd_aux             = 'RD21',
  rd_aux2            = 'RD12', # Corregido
  mora_temp          = 'MORA_30_3',

  # Configuración de proyecciones por cuentas
  proy_def_temp_xc   = 'All_proy', #All_proy met1_xc met2_xc met3_xc
  suav_def_temp_xc   = 1,
  proy_mor_temp_xc   = 'All_proy', #All_proy det1_xm det2_xm det3_xm
  suav_mor_temp_xc   = 1,
  prof_hist_xc       = 14,

  # Configuración de proyecciones por montos
  proy_def_temp_xm   = "All_proy", #All_proy met1_xm met2_xm met3_xm
  suav_def_temp_xm   = 1,
  proy_mor_temp_xm   = 'All_proy', #All_proy det1_xm det2_xm det3_xm
  suav_mor_temp_xm   = 1,
  prof_hist_xm       = 14,
  #fact_to_proy_xc    = [fc_1, fc_2],
  #fact_to_proy_xm    = [fm_1, fm_2],

  # Detalles del gráfico de calibración
  titulo             = titulo,
  pos_leyenda        = (0.5, -0.25),
  eje_y_xc           = 0.25,
  eje_y_xm           = 0.25,
  dim_grafico        = (25, 6.5),
  punt_mora          = 85, #Tamaño de puntos de mora temprana
  etiquetas          =True,
  pos_etiquetas_xm   =10,
  pos_etiquetas_xc   =10,
  tamaño_etiqueta    =20,

  # Exportar factores para vista driver
  exportar_factores  = True,
  mat_porc=-24
)

# COMMAND ----------

# MAGIC %md
# MAGIC ####3. Vista Troncales

# COMMAND ----------

gini = Obj_Seg.EvolDisVar(
  # Parametros de figura individual
    query_filtro = '',
    codmes_ini   = 202201,
    codmes_fin   = 202312,
    amplitud     = 24,
    disc_target  = [24, 21,18],
    umbral_sup   = 0.4,
    umbral_inf   = 0.3,
    umbral_opc   = 0.1,
    moviles      = True,
    dim_grafico  = (15, 6),
    pos_leyenda  = (0.5, -0.3),
 
    etiquetas    = True,
    tamaño_etiqueta    =12,

    # Parametros de figura final
    ncolumnas = 3,
    nfilas    = 2,
    variables = xb_mod,
    dim_grafico_total = (25, 10),
    ncol_leyenda = 5,
    vspacio=-0.25,
    hspacio=4
  )

# COMMAND ----------

Obj_Seg.base_monitoreo.groupBy("INGRESO_SOL_CONY_TP_Q").count().show()

# COMMAND ----------

fc

# COMMAND ----------

caldri = Obj_Seg.CalVar(
  # Parametros de figura individual
  query_uso          = f"(CODMES BETWEEN 202012 AND 202302) AND CODMES NOT IN (202201,202208)",
  pd_seg             = 'PD_GAHI',
  rd_aux             = 'RD12',
  pd_comparacion     = False, #Para tomar mas de un PD
  factores_cuentas   = fc,
  factores_montos    = fm,
  mat_importancia    = 5, # minimo % de materialidad por bucket mostrado
  cast_int           = True, #para los flag los cambia a entero
  dim_grafico        = (25, 6),
  pos_leyenda        = (0.5, -0.25),
  punt_mora          = 70, # tamaño max 1000
 
  # Parametros de figura final
  tipo_vista = 'M',
  ncolumnas = 3,
  nfilas    = 2,
  variables = var_mod,
  dim_grafico_total = (25, 10),
  ncol_leyenda     = 5,
  etiquetas_total=True, 
  pos_etiqueta      = 0.05,
  vspacio           = -0.5,
  hspacio           = 1,
  filtro_var="not like '%issing%'"# para no mostrar missing,
)

# COMMAND ----------

caldri = Obj_Seg.CalVar(
  # Parametros de figura individual
  query_uso          = f"(CODMES BETWEEN 202012 AND 202302) AND CODMES NOT IN (202201,202208)",
  pd_seg             = 'PD_GAHI',
  rd_aux             = 'RD12',
  pd_comparacion     = True, #Para tomar mas de un PD
  factores_cuentas   = fc,
  factores_montos    = fm,
  mat_importancia    = 5, # minimo % de materialidad por bucket mostrado
  cast_int           = True, #para los flag los cambia a entero
  dim_grafico        = (25, 6),
  pos_leyenda        = (0.5, -0.25),
  punt_mora          = 70, # tamaño max 1000
 
  # Parametros de figura final
  tipo_vista = 'C',
  ncolumnas = 3,
  nfilas    = 2,
  variables = var_mod,
  dim_grafico_total = (25, 10),
  ncol_leyenda     = 5,
  etiquetas_total=True, 
  pos_etiqueta      = 0.05,
  vspacio           = -0.5,
  hspacio           = 1,
  filtro_var="not like '%issing%'"# para no mostrar missing
)

# COMMAND ----------

psi_dri = Obj_Seg.MixPSIVar(
  # Parametros de figura individual
    query_uso='CODMES >= 202101',
    cast_int=True,
    codmes_inicio1=202101,
    codmes_fin1=202112,
    codmes_inicio2=202301,
    codmes_fin2=202312,
    dim_grafico=(25, 6),
    pos_leyenda=(0.5, -0.25),
 
    # Parametros de figura final
    tipo_vista = 'M',
    ncolumnas = 3,
    nfilas    = 2,
    variables = var_mod,
    dim_grafico_total = (25, 10),
    ncol_leyenda=5,
    tamaño_leyenda = 8.25,
    vspacio=-0.25,
    hspacio=0
)

# COMMAND ----------

gini = Obj_Seg.EvolDisVar(
  # Parametros de figura individual
    query_filtro = '',
    codmes_ini   = 202110,
    codmes_fin   = 202312,
    amplitud     = 24,
    disc_target  = [24, 21,18],
    umbral_sup   = 0.15,
    umbral_inf   = 0.05,
    #umbral_opc   = 0.1,
    moviles      = True,
    dim_grafico  = (15, 6),
    pos_leyenda  = (0.5, -0.3),
 
    etiquetas    = True,
    tamaño_etiqueta    =12,

    # Parametros de figura final
    ncolumnas = 3,
    nfilas    = 2,
    variables = var_mod_st,
    dim_grafico_total = (25, 10),
    ncol_leyenda = 5,
    vspacio=-0.25,
    hspacio=4
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ####4. Vista Power BI

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.1. Vista Factores

# COMMAND ----------

import pandas as pd

def procesar_factores(n_SegProy, Mes_Def):
    df_factores = pd.DataFrame()  # Inicializar un DataFrame vacío
    
    for i in range(1, n_SegProy + 1):
        # Generar los nombres de los DataFrames dinámicamente
        fc = globals()[f'fc_{i}']
        fm = globals()[f'fm_{i}']
        
        # Añadir la columna SEG_PROYECCION
        fc['SEG_PROYECCION'] = i
        fc['CODMES_DEF'] = Mes_Def

        # Realizar la unión de los DataFrames
        df_resultante = pd.merge(fc, fm, on="CODMES", how="inner", suffixes=('_fc', '_fm'))

        # Seleccionar las columnas deseadas
        df_resultante = df_resultante[['CODMES_DEF','CODMES', 'SEG_PROYECCION', 'RD_PROY_DEF_fm', 'FAC_CAL_DRI_M', 'RD_PROY_DEF_fc', 'FAC_CAL_DRI_N']]

        # Reemplazar los valores de factores NaN por 0 y RD por 0.005
        df_resultante[['FAC_CAL_DRI_M', 'FAC_CAL_DRI_N']] = df_resultante[['FAC_CAL_DRI_M', 'FAC_CAL_DRI_N']].fillna(0)
        df_resultante[['RD_PROY_DEF_fm', 'RD_PROY_DEF_fc']] = df_resultante[['RD_PROY_DEF_fm', 'RD_PROY_DEF_fc']].fillna(0.005)

        # Cambiar los nombres de las columnas
        df_resultante.rename(columns={'RD_PROY_DEF_fm': 'proyeccion_montos'}, inplace=True)
        df_resultante.rename(columns={'FAC_CAL_DRI_M': 'factor_montos'}, inplace=True)
        df_resultante.rename(columns={'RD_PROY_DEF_fc': 'proyeccion_cuentas'}, inplace=True)
        df_resultante.rename(columns={'FAC_CAL_DRI_N': 'factor_cuentas'}, inplace=True)

        # Concatenar el DataFrame resultante al DataFrame final
        df_factores = pd.concat([df_factores, df_resultante], ignore_index=True)
    return df_factores

# COMMAND ----------

df_factores=procesar_factores(1,202501)

# COMMAND ----------

df_factores

# COMMAND ----------

# Convertir el DataFrame de Pandas a un DataFrame de Spark
df_factores = spark.createDataFrame(df_factores)
df_factores.createOrReplaceTempView("temp_view")
spark.sql("CREATE OR REPLACE TABLE catalog_lhcl_prod_bcp.bcp_edv_fabseg.T45988_FAC_HM_ADM_HIP_GAHI_202502_F LOCATION 'abfss://bcp-edv-fabseg@adlscu1lhclbackp05.dfs.core.windows.net/HIPOTECARIO_2025/APP/202502/T45988_FAC_HM_ADM_HIP_GAHI_202502_F' AS SELECT * FROM temp_view")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.2. Vista Cubo 

# COMMAND ----------

pd_vig='PD_GAHI' 
pd_cal='PD_GAHI'
pd_trc='XB_GAHI_PRELIMINAR'
tipo_banda = 'Jeffrey'#Jeffrey Vasiseck
codmes_default = 202301
columna_monto='MTOAPROBADO_SOLES'
titulo='Calibración Applicant App Garantia Hipotecaria'

# COMMAND ----------

from pyspark.sql import SparkSession
base_seguimiento = spark.sql("select distinct * from catalog_lhcl_prod_bcp.bcp_edv_fabseg.T45988_HM_ADM_HIP_GAHI_202502_F")

# COMMAND ----------

base_seguimiento = base_seguimiento.filter("CODMES>=202101 AND DEF_120_BLOQ_REF_0 = 0 AND PD_GAHI IS NOT NULL")

# COMMAND ----------

Obj_Seg.base_monitoreo = Obj_Seg.base_monitoreo.withColumn("SEG_PROYECCION", lit(1))

# COMMAND ----------

monto = "MTOAPROBADO_SOLES"

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.3. Formación Cubo 

# COMMAND ----------

from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Definir la fecha base
MES_DEF_ACT = 202501
default = 24

# Convertir la fecha base a un objeto datetime
# base_date = datetime.strptime(MES_DEF_ACT, "%Y%m")

# Función para calcular los meses anteriores
def calculate_previous_months(codmes, cant):
  numeromes = (codmes // 100) * 12 + (codmes % 100)
  return ((numeromes - 1 + cant) // 12) * 100 + ((numeromes + 11 - 11 * cant) % 12) + 1

# Calcular los meses anteriores

codmes_a1 = calculate_previous_months(MES_DEF_ACT, -default+11)
codmes_a2 = calculate_previous_months(MES_DEF_ACT, -default+10)
codmes_a3 = calculate_previous_months(MES_DEF_ACT, -default+9)
codmes_a4 = calculate_previous_months(MES_DEF_ACT, -default+8)
codmes_a5 = calculate_previous_months(MES_DEF_ACT, -default+7)
codmes_a6 = calculate_previous_months(MES_DEF_ACT, -default+6)
codmes_a7 = calculate_previous_months(MES_DEF_ACT, -default+5)
codmes_a8 = calculate_previous_months(MES_DEF_ACT, -default+4)
codmes_a9 = calculate_previous_months(MES_DEF_ACT, -default+3)
codmes_a10 = calculate_previous_months(MES_DEF_ACT, -default+2)
codmes_a11 = calculate_previous_months(MES_DEF_ACT, -default+1)
codmes_a12 = calculate_previous_months(MES_DEF_ACT, -default)


# COMMAND ----------

# Cargar las tablas para el cruce
df_a = Obj_Seg.base_monitoreo
df_b = spark.table("catalog_lhcl_prod_bcp.bcp_edv_fabseg.T45988_FAC_HM_ADM_HIP_GAHI_202502_F")
# Crear la columna ult_rd
df_a = df_a.withColumn("ult_rd", when(col("CODMES") == codmes_a11, col("DEF23"))
                                 .when(col("CODMES") == codmes_a10, col("DEF22"))
                                 .when(col("CODMES") == codmes_a9, col("DEF22"))
                                 .when(col("CODMES") == codmes_a8, col("DEF21"))
                                 .when(col("CODMES") == codmes_a7, col("DEF20"))
                                 .when(col("CODMES") == codmes_a6, col("DEF19"))
                                 .when(col("CODMES") == codmes_a5, col("DEF18"))
                                 .when(col("CODMES") == codmes_a4, col("DEF17"))
                                 .when(col("CODMES") == codmes_a3, col("DEF16"))
                                 .otherwise(0))

# COMMAND ----------

df_a.columns

# COMMAND ----------

# Realizar la unión izquierda
df_joined = df_a.join(df_b, (df_a.CODMES == df_b.CODMES) & (df_a.SEG_PROYECCION == df_b.SEG_PROYECCION), "left")

# Calcular las nuevas columnas
df_cal01 = df_joined.select(
    df_a["*"],
    df_b.factor_montos,
    df_b.factor_cuentas,
    (col("ult_rd") * df_b.factor_montos).alias("def12_proy_factor_m"),
    (col("ult_rd") * df_b.factor_cuentas).alias("def12_proy_factor_n"),
    df_b.proyeccion_montos.alias("def12_proy_valor_m"),
    df_b.proyeccion_cuentas.alias("def12_proy_valor_n")
)

# la última tabla creada aquí es df_cal01

# COMMAND ----------

df_cal01.columns

# COMMAND ----------

#CUBO PRINCIPAL

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when, count

# Crear las columnas flgventana_diagnostico, flgventana_reciente, flgdef_cerrado, flgdef_temp, flgmora_temp
df_cal01 = df_cal01.withColumn("flgventana_diagnostico", when(col("CODMES").between(codmes_a11, codmes_a6), 1).otherwise(0)) \
           .withColumn("flgventana_reciente", when(col("CODMES").between(codmes_a3, codmes_a1), 1).otherwise(0)) \
           .withColumn("flgdef_cerrado", when(col("CODMES") <= codmes_a12, 1).otherwise(0)) \
           .withColumn("flgdef_temp", when(col("CODMES") <= codmes_a9, 1).otherwise(0)) \
           .withColumn("flgmora_temp", when(col("CODMES") <= 202401, 1).otherwise(0)) \
           .withColumn("PASA_PAUTA", when((col("SC_GAHI") >= 345) & (col("SEGMENTO_BANCA_F") == "1. Afluente"), 1)
                                   .when((col("SC_GAHI") >= 415) & (col("SEGMENTO_BANCA_F") == "2. Consumo"), 1)
                                   .when((col("SC_GAHI") >= 415) & (col("SEGMENTO_BANCA_F") == "3. Resto"), 1)
                                   .otherwise(0)) 

# COMMAND ----------

#Creación de nuevas PDs
# # Calcular PD_TRAD para las columnas XB especificadas y renombrar las columnas resultantes
xb_columns = {
    'XB_GAHI_PRELIMINAR': 'PD_TRC',
    'XB_NUE_APP_CEF': 'PD_NUEVA',
    'XB_MODUL_GAHI': 'PD_MODULO_GAHI'
}

for col_name, new_col_name in xb_columns.items():
    df_cal01 = df_cal01.withColumn(new_col_name, expr(f"1 / (1 + exp(-{col_name}))"))
    

# COMMAND ----------

sum(col("PD_GAHI") * col("MTOAPROBADO_SOLES")).alias("pd_01_xm"),
    sum(col("PD_GAHI") * col("MTOAPROBADO_SOLES")).alias("pd_02_xm"),
    sum(col("PD_TRC") * col("MTOAPROBADO_SOLES")).alias("pd_03_xm"),
    sum(col("PD_NUEVA") * col("MTOAPROBADO_SOLES")).alias("pd_04_xm"),
    sum(col("PD_MODULO_GAHI") * col("MTOAPROBADO_SOLES")).alias("pd_05_xm"),
    sum(col("PD_GAHI") * col("MTOAPROBADO_SOLES")).alias("pd_06_xm"),
    sum(col("PD_GAHI") * col("MTOAPROBADO_SOLES")).alias("pd_07_xm"),
    sum(col("PD_GAHI") * col("MTOAPROBADO_SOLES")).alias("pd_08_xm"),
    sum(col("PD_GAHI") * col("MTOAPROBADO_SOLES")).alias("pd_09_xm"),
    sum(col("PD_GAHI") * col("MTOAPROBADO_SOLES")).alias("pd_10_xm"),

# COMMAND ----------

# LISTA DE NOMBRES DE COLUMNAS
lista_pd = [
    "PD_GAHI",
    "PD_GAHI",
    "PD_TRC",
    "PD_NUEVA",
    "PD_MODULO_GAHI",
    "PD_GAHI",
    "PD_GAHI",
    "PD_GAHI",
    "PD_GAHI",
    "PD_GAHI"
]
# Crear una lista de expresiones de agregación
pd_result_xm = [sum(col(c) * col(monto)).alias(f"pd_{str(i+1).zfill(2)}_xm") for i, c in enumerate(lista_pd)]
# Crear una lista de expresiones de agregación sin multiplicar por monto
pd_result_xn = [sum(col(c)).alias(f"pd_{str(i+1).zfill(2)}_xn") for i, c in enumerate(lista_pd)]



# COMMAND ----------

#DEFINIR LISTA DE EJES 
lista_ejes = {
    "CAMPANIA_AGRUPADA": "EJE_P1",
    "TIPO_FONDEO": "EJE_P2",
    "SEGMENTO_BANCA_F": "EJE_P3",  
    "RANGO_MTOVIVI_F": "EJE_P4",
    "CTDPLAZOAPROBADO": "EJE_P5",
    "PD_GAHI_Q": "EJE_S1",
    "PD_GAHI_PRELIMINAR_Q": "EJE_S2",
    "FLG_MONTO_CAL": "EJE_S3",
    "codyear": "EJE_S4",
    "INGRESO_SOL_CONY_Q": "EJE_S5",
    "RIESGO_ZONA2": "EJE_S6",
    "MONTO_R": "EJE_S7",
    "RAN_PLAZO": "EJE_S8",
    "TASA_HIP_Q": "EJE_S9",
    "TIPO_EMPLEO_F": "EJE_S10",
    "EDAD_R": "EJE_S11",
    "MAX_GRADO": "EJE_S12",
    "LTV_CORREGIDO_Q": "EJE_S13",
    "INGRESO_RBM_Q": "EJE_S14",
    "MARCA_HML_T": "EJE_S15",
    "PASA_PAUTA": "EJE_S16"
}

# Crear una lista de expresiones de selección usando el diccionario
eje_result = [col(k).alias(v) for k, v in lista_ejes.items()]

# COMMAND ----------

# AGRUPAR Y AGREGAR DATOS
cubo_principal = df_cal01.groupBy(
    "CODMES",
    "seg_proyeccion",
    "flgventana_diagnostico",
    "flgventana_reciente",
    "flgdef_cerrado",
    "flgdef_temp",
    "flgmora_temp",
    col("SEGMENTO_BANCA_F").alias("EJE_TOOLTIP"),
    *eje_result,
).agg(
    *pd_result_xm,
    sum(col(monto)).alias("m"),
    sum(when(col("CODMES") > codmes_a12, 0).otherwise(col("DEF24") * col(monto))).alias("def_cerrado_xm"),
    sum(when(col("CODMES") > codmes_a9, 0).otherwise(col("DEF21") * col(monto))).alias("def_temp_xm"),
    sum(when(col("CODMES") > 202401, 0).otherwise(col("DEF12") * col(monto))).alias("mora_temp_xm"),
    sum(when(col("CODMES").between(codmes_a11, codmes_a6), col("def12_proy_factor_m") * col(monto)).otherwise(0)).alias("def_proy_factor_dt_xm"),
    sum(when(col("CODMES").between(codmes_a5, codmes_a3), col("def12_proy_factor_m") * col(monto)).otherwise(0)).alias("def_proy_factor_mt_xm"),
    sum(when(col("CODMES").between(codmes_a11, codmes_a6), col("def12_proy_valor_m") * col(monto)).otherwise(0)).alias("def_proy_valor_dt_xm"),
    sum(when(col("CODMES").between(codmes_a5, codmes_a3), col("def12_proy_valor_m") * col(monto)).otherwise(0)).alias("def_proy_valor_mt_xm"),
    *pd_result_xn,
    count(lit(1)).alias("n"),
    sum(when(col("CODMES") > codmes_a12, 0).otherwise(col("DEF24"))).alias("def_cerrado_xn"),
    sum(when(col("CODMES") > codmes_a9, 0).otherwise(col("DEF21"))).alias("def_temp_xn"),
    sum(when(col("CODMES") > 202401, 0).otherwise(col("DEF12"))).alias("mora_temp_xn"),
    sum(when(col("CODMES").between(codmes_a11, codmes_a6), col("def12_proy_factor_n")).otherwise(0)).alias("def_proy_factor_dt_xn"),
    sum(when(col("CODMES").between(codmes_a5, codmes_a3), col("def12_proy_factor_n")).otherwise(0)).alias("def_proy_factor_mt_xn"),
    sum(when(col("CODMES").between(codmes_a11, codmes_a6), col("def12_proy_valor_n")).otherwise(0)).alias("def_proy_valor_dt_xn"),
    sum(when(col("CODMES").between(codmes_a5, codmes_a3), col("def12_proy_valor_n")).otherwise(0)).alias("def_proy_valor_mt_xn")
)

cubo_principal_final = cubo_principal.toPandas()

# COMMAND ----------

# PASAR DE DATAFRAME A UNA TABLA DE DATABRICKS
cubo_principal.createOrReplaceTempView("temp_view")

spark.sql("""
CREATE OR REPLACE TABLE catalog_lhcl_prod_bcp.bcp_edv_fabseg.cubo_gahi 
LOCATION 'abfss://bcp-edv-fabseg@adlscu1lhclbackp05.dfs.core.windows.net/HIPOTECARIO_2025/APP/202502/cubo_gahi' 
AS SELECT * FROM temp_view
""")

# COMMAND ----------

#ETIQUETA DE EJES

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType


# Definir el esquema
schema = StructType([
    StructField("EJE", StringType(), True),
    StructField("ETIQUETA", StringType(), True)
])

# Crear los datos
data = [
    ("EJE_P1", "Producto"),
    ("EJE_P2", "Tipo de Fondeo"),
    ("EJE_P3", "Segmento Banca"),
    ("EJE_P4", "Monto de Vivienda"),
    ("EJE_P5", "Cant Plazo Aprobado"),
    ("EJE_S1", "Rango de PD"),
    ("EJE_S2", "Score Troncal Garantía Hipotecaria"),
    ("EJE_S3", "Flag de monto aprobado (S/)>= 325k"),
    ("EJE_S4", "Vista anual"),
    ("EJE_S5", "Ingreso Titular + Conyuge"),
    ("EJE_S6", "Riesgo Zona"),
    ("EJE_S7", "Monto aprobado"),
    ("EJE_S8", "Rango Plazo"),
    ("EJE_S9", "Rango de Tasa"),
    ("EJE_S10", "Tipo de Empleo"),
    ("EJE_S11", "Rango de Edad"),
    ("EJE_S12", "Max Grado"),
    ("EJE_S13", "LTV"),
    ("EJE_S14", "Rango de Ingreso Titular"),
    ("EJE_S15", "Marca HML"),
    ("EJE_S16", "Pasa Pauta")
]

# Crear el DataFrame
df_etiquetas_ejes = spark.createDataFrame(data, schema)

df_etiquetas_ejes_final = df_etiquetas_ejes.toPandas()


# COMMAND ----------


#ETIQUETA DE PDs

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

# Crear una sesión de Spark
spark = SparkSession.builder.appName("SQLtoPySpark").getOrCreate()

# Definir el esquema
schema = StructType([
    StructField("PD", StringType(), True),
    StructField("DESCRIPTIVO", StringType(), True)
])

# Crear los datos
data = [
    ("01", "01-PD Vigente"),
    ("02", "02-PD Calibrado"),
    ("03", "03-PD Troncal"),
    ("04", "04-PD Nueva"),
    ("05", "05-PD Modulo Gahi"),
    ("06", "06-PD Demográfico"),
    ("07", "07-PD Activo"),
    ("08", "08-PD Transaccional"),
    ("09", "09-PD Pasivo"),
    ("10", "10-PD RCC"),
]

# Crear el DataFrame
df_descriptivo_pds = spark.createDataFrame(data, schema)
# Convertir a Pandas
df_descriptivo_pds_final = df_descriptivo_pds.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.4. Exportar CSV

# COMMAND ----------

#EXPORTAR EN ARCHIVO CSV (PARA EL CUBO_PRINCIPAL SE ESTÁ EXPORTANDO EN ZIP (COMPRIMIDO))

import zipfile

# Guardar los DataFrames como archivos CSV
df_etiquetas_ejes_final.to_csv("/Workspace/Users/mariapaloma@bcp.com.pe/App_Gahi/etiquetas_ejes.csv", index=False)
df_descriptivo_pds_final.to_csv("/Workspace/Users/mariapaloma@bcp.com.pe/App_Gahi/descriptivo_pds.csv", index=False)

# Guardar y comprimir cubo_principal_final
csv_path = "/Workspace/Users/mariapaloma@bcp.com.pe/App_Gahi/cubo_principal.csv"
zip_path = "/Workspace/Users/mariapaloma@bcp.com.pe/App_Gahi/cubo_principal.zip"

cubo_principal_final.to_csv(csv_path, index=False)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(csv_path, arcname="cubo_principal.csv")

# COMMAND ----------

df_cal01.columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. Vista 2x2

# COMMAND ----------

drivers_s1 = ['XB_AD_HOC_2','XB_DEM_S_ADCW_2','XB_ADC_POND85_F','XB_HIP','XB_AD_HOC','XB_DEM_S_ADCW','XB_ACT_S_ADCW','XB_TRX_S_ADCW','XB_PAS_S_ADCW','XB_RCC_S_ADCW','PD_APP_HIP_TRDPYME_2Q24','PD_APP_TRAD']

hm_dis_gini_dri, hm_dis_ks_dri = Obj_Seg.DisVar(
  query_filtro    = "CODMES>=202101 AND DEF_120_BLOQ_REF_0 = 0",
  variables       = drivers_s1,
  target          = 24,
  codmes_inicio   = 202201,
  codmes_final    = 202211,
  amplitud        = 3,
  moviles         = False
)
displayHTML(hm_dis_gini_dri.to_html(escape=False))

# COMMAND ----------

VARIABLE = 'FLG_MONTO_CAL'
caldri_evol = Obj_Seg.EvolCalDri(
  variable = VARIABLE,
  # Configuración de filtro de uso y pd de calibración
  query_uso          = f"CODMES >= 202101 AND {VARIABLE} NOT LIKE '%ssing%'",
  pd_seg             = pd_vig,
  pd_comparacion     = True,
  proys_def_temp     = 6,
  proys_mor_temp     = 0,
  rd_aux             = 'RD21',
  rd_aux2            = 'RD12',
  mora_temp          = 'MORA_30_3',
  amplitud           = 1,
  moviles            = False,
  factor_cuentas     = fc,
  factor_montos      = fm,
  pos_leyenda        = (0.5, -0.35),
  dim_grafico        = (25, 6.5),
  punt_mora          = 85,
  missing_bucket     = False,
 
  # Parametros de figura final
  tipo_vista         = 'M',
  ncolumnas          = 2,
  nfilas             = 2,
  dim_grafico_total  = (22, 12),
  ncol_leyenda       = 7,
  tamaño_leyenda     = 8.5,
  etiquetas_total    = True,
  pos_etiqueta       = 0,
  vspacio            = -0.5,
  hspacio            = 0.5
)

