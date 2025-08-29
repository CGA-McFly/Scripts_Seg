# Databricks notebook source
# MAGIC %md
# MAGIC ## Ficha de Seguimiento
# MAGIC  
# MAGIC #### Proyecto: SEGUIMIENTO APPLICANT TRADICIONAL
# MAGIC **Tablas Fuentes:** catalog_lhcl_prod_bcp.bcp_edv_fabseg.T45988_HM_ADM_HIP_TRAD_202503_F
# MAGIC •catalog_lhcl_prod_bcp.bcp_edv_fabseg.T45988_HM_MTZ_ADM_HIPOTECARIO_DRIVER: Universo Applicant
# MAGIC •catalog_lhcl_prod_bcp.bcp_edv_fabseg.t39290_MD_RELCLIENTEDL: Llave relacionadora a tablas del LHCL
# MAGIC •catalog_lhcl_prod_bcp.bcp_ddv_adrmmgr_seginfobasesgenerales_vu.mm_portafoliocredito: Tabla de default actualizado
# MAGIC •catalog_lhcl_prod_bcp.bcp_edv_fabseg.t39290_de_hm_sitiacion_laboral_experian: Situación laboral experian
# MAGIC •catalog_lhcl_prod_bcp.bcp_edv_fabseg.T45988_EXT_SEG_AGRUP_LTV : Tabla LTV
# MAGIC •catalog_lhcl_prod_bcp.bcp_edv_fabseg.T45988_HM_SOLICITUDES_CH_RMS_ESTADOS : Tabla Suite – Estados
# MAGIC •catalog_lhcl_prod_bcp.bcp_edv_fabseg.T45988_GREMIO_202503 : Tabla Gremio
# MAGIC •catalog_lhcl_prod_bcp.bcp_edv_fabseg.T39290_hm_mv_1_5_2 : Tabla Marca Vulnerable
# MAGIC •catalog_lhcl_prod_bcp.bcp_edv_fabseg.T45988_INCIDENCIAS_CEF_202503: Tabla Flag Problemas CEF
# MAGIC •catalog_lhcl_prod_bcp.bcp_udv_int_vu.m_cuentacreditopersonal: Tabla para Tasa y Cuota
# MAGIC •catalog_lhcl_prod_bcp.bcp_ddv_rbmrbmper_scorecomportamiento_vu.mm_incumplimientoclienteriesgos: Tabla Default CEF y TC
# MAGIC •catalog_lhcl_prod_bcp.bcp_udv_int_vu.m_solicitudcreditohipotecario: Fuente Créditos de solicitud hipotecario información desde 202308- CSOL
# MAGIC
# MAGIC •catalog_lhcl_prod_bcp.bcp_ddv_rbmrbmper_solicitudch_vu.hm_solicitudcreditohipotecariorbm: Nueva Fuente RBM productiva
# MAGIC •catalog_lhcl_prod_bcp.bcp_ddv_rbmrbmper_reportlogevaluacion_vu.hd_personasolicitudevaluacioncreditohipoticariorbmper: Tabla Suite información desde 202308
# MAGIC •catalog_lhcl_prod_bcp.bcp_edv_fabseg.T45988_HM_EXCEPCIONES_HIPOTECARIO_202503: Tabla Excepciones
# MAGIC •	catalog_lhcl_prod_bcp.bcp_edv_fabseg.T45988_MM_APPSCORE_HIP_TRD_202503: Base de variables troncales
# MAGIC
# MAGIC  
# MAGIC #### Objetivo:
# MAGIC Realizar el seguimiento del modelo Applicant tradicional a nivel de clientes Banca Personas
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
# MAGIC | 4      | Lesly Maza  | Gerardo Soto        | 15/03/2025  | Migración a Data Bricks  |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Seguimiento Applicant Tradicional

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
from pyspark.sql.types import StructType, StructField, StringType

# COMMAND ----------

#Mostrar los duplicados
def print_res(sparkf_df):
  import pandas as pd
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)
  pd_spark_df = sparkf_df.toPandas()
 
  return pd_spark_df

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

# MAGIC %md
# MAGIC ####Universo

# COMMAND ----------

from pyspark.sql import SparkSession
base_seguimiento = spark.sql("select distinct * from catalog_lhcl_prod_bcp.bcp_edv_fabseg.T45988_HM_ADM_HIP_TRAD_202503_F")

# COMMAND ----------

def print_res(sparkf_df):
    import pandas as pd
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd_spark_df = sparkf_df.toPandas()
    return pd_spark_df

# COMMAND ----------


base_seguimiento = base_seguimiento.withColumn("CODMES", col("CODMES").cast("integer"))
base_seguimiento = base_seguimiento.withColumn("DEF5", when(col("DISTANCIA_DEF24") <= 5, lit(1)).otherwise(lit(0)))
base_seguimiento = base_seguimiento.withColumn("DEF4", when(col("DISTANCIA_DEF24") <= 4, lit(1)).otherwise(lit(0)))  


# COMMAND ----------

base_seguimiento = base_seguimiento.withColumn(
    "INGRESO_CONY_TIT",
    coalesce(col("INGRESO_SOL_CONY"), expr("0")) + coalesce(col("INGRESO_RBM"), expr("0"))
)

# COMMAND ----------

base_seguimiento = base_seguimiento.withColumn("CAT_ZONA", 
                   when(col("CAT_ZONA1").isNull(), "99. Missing")
                   .when(col("CAT_ZONA1").contains("LIMA"), "Lima")
                   .otherwise("Provincia"))

# COMMAND ----------

base_seguimiento = base_seguimiento.withColumn(
    "CSOL_XB_RBM", (174.25 - F.col("CSOL_SC_RBM")) / 57.708
)

base_seguimiento = base_seguimiento.withColumn("CSOL_PD_RBM", 1 / (1 + F.exp(-F.col("CSOL_XB_RBM"))))

base_seguimiento = base_seguimiento.withColumn("PD_RBM", coalesce(col("PD_ESTADOS"), col("CSOL_PD_RBM")))

# COMMAND ----------

base_seguimiento = base_seguimiento.withColumn("LTV_CORREGIDO_VF", when(col("CODMES") >= 202308,  col("NFT_LTV")).otherwise(col("LTV_CORREGIDO_0")))

# COMMAND ----------

num_filas = base_seguimiento.count()
num_columnas = len(base_seguimiento.columns)

print(f"Número de filas: {num_filas}")
print(f"Número de columnas: {num_columnas}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 0. Definicion de Seguimiento

# COMMAND ----------

pd_rbm='PD_RBM'
pd_vig='PD_APP_TRAD' 
pd_cal='PD_APP_HIP_TRDPYME_2Q24'
pd_trc='XB_ADC_POND85_F'
tipo_banda = 'Jeffrey'#Jeffrey Vasiseck
codmes_default = 202302
columna_monto='MTOAPROBADO_SOLES'
titulo='Calibración Applicant App Tradicional'

# COMMAND ----------

condiciones= "CODMES >= 201901 AND DEF_120_BLOQ_REF_0 = 0"

# COMMAND ----------

# Objeto de Seguimiento
Obj_Seg = rmm.MonitorScore_v01(
  nombre_modelo         = 'Personas Modelo Applicant Tradicional',
  codigo_modelo         = 'MOD-BCP-20658',
  tier                  = 'II',
  detalle_seguimiento   = 'Seguimiento Applicant Tradicional',
  mes_seguimiento       = '202503',
  base_monitoreo        = base_seguimiento,
  pd1                   = pd_vig,
  pd2                   = pd_cal,
  pd3                   = pd_rbm,
  monto_credito         = columna_monto,
  query_univ            = "DEF_120_BLOQ_REF_0=0 AND PD_APP_HIP_TRDPYME_2Q24 IS NOT NULL",
  bandas                = tipo_banda,
  codmes_default        = codmes_default,
  meses_ventana_target  = 24,
  meses_atraso_target   = 4
)

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

# DBTITLE 1,missing pd vig
result = base_seguimiento.where(col("CODMES")>=202101).groupBy("CODMES").agg(
    F.sum(F.when(F.col("PD_APP_TRAD").isNull(), 1).otherwise(0)).alias("MISS_PD"),
    F.sum(F.when(F.col("PD_APP_TRAD").isNull() & F.col("CODSOLICITUD").isNull(), 1).otherwise(0)).alias("MISS_SOL"),
    F.count("*").alias("N")
).orderBy("CODMES")

result.show(200)

# COMMAND ----------

# DBTITLE 1,missing pd propuesto
result = base_seguimiento.where(col("CODMES")>=202101).groupBy("CODMES").agg(
    F.sum(F.when(F.col("PD_APP_HIP_TRDPYME_2Q24").isNull(), 1).otherwise(0)).alias("MISS_PD"),
    F.sum(F.when(F.col("PD_APP_HIP_TRDPYME_2Q24").isNull() & F.col("CODSOLICITUD").isNull(), 1).otherwise(0)).alias("MISS_SOL"),
    F.count("*").alias("N")
).orderBy("CODMES")

result.show(200)

# COMMAND ----------

# DBTITLE 1,Validacion: Base seguimiento
df_filtered = base_seguimiento.filter(
    (col("CODMES") >= 202201) &
    (col('DEF_120_BLOQ_REF_0')==0) &
    (col(pd_cal).isNotNull())
)

cubo1 = df_filtered.groupBy("CODMES").agg(
    F.count("*").alias("N"),
    (F.sum(columna_monto) / 1000000).alias("MONTO"),
    (F.sum(pd_vig) / F.count("*")).alias("PD_VIG"),
    (F.sum("DEF24") / F.count("*")).alias("RD24"),
    (F.sum("DEF18") / F.count("*")).alias("RD18"),
    (F.sum("DEF24") ).alias("DEF24")
)

# Mostrar el DataFrame resultante
cubo1.orderBy('CODMES').show(60)

# COMMAND ----------

df_filtered = base_seguimiento.filter(
    (col("CODMES") >= 202201) &
    (col('DEF_120_BLOQ_REF_0')==0) &
    (col(pd_rbm).isNotNull())
)

cubo1 = df_filtered.groupBy("CODMES").agg(
    F.count("*").alias("N"),
    (F.sum(columna_monto) / 1000000).alias("MONTO"),
    (F.sum(pd_vig) / F.count("*")).alias("PD_VIG"),
    (F.sum("DEF24") / F.count("*")).alias("RD24"),
    (F.sum("DEF18") / F.count("*")).alias("RD18"),
    (F.sum("DEF24") ).alias("DEF24")
)

# Mostrar el DataFrame resultante
cubo1.orderBy('CODMES').show(60)

# COMMAND ----------

# DBTITLE 1,Validacion: Base sin filtro
# Crear el DataFrame CUBO2 (similar al CUBO1 pero con otra fuente de datos)
df_filtered2 = base_seguimiento.filter(
    (col("CODMES") >= 202201) &
    (col('DEF_120_BLOQ_REF_0')==0)
)

cubo2 = df_filtered2.groupBy("CODMES").agg(
    F.count("*").alias("N"),
    (F.sum(columna_monto) / 1000000).alias("MONTO"),
    (F.sum(pd_vig) / F.count("*")).alias("PD_VIG"),
    (F.sum("DEF24") / F.count("*")).alias("RD24"),
    (F.sum("DEF18") / F.count("*")).alias("RD18"),
    (F.sum("DEF24") ).alias("DEF24")
)

# Mostrar el DataFrame resultante
cubo1.orderBy('CODMES').show(60)


# COMMAND ----------

cubo2_renamed = cubo2.select(
    F.col("CODMES"),
    F.col("N").alias("N_2"),
    F.col("MONTO").alias("MONTO_2"),
    F.col("PD_VIG").alias("PD_VIG_2"),
    F.col("RD24").alias("RD24_2"),
    F.col("RD18").alias("RD18_2"),
    F.col("DEF24").alias("DEF24_2")
)
cubo_combined = cubo1.join(cubo2_renamed, on="CODMES", how="inner")

# Calcular la diferencia de cada campo
cubo_diff = cubo_combined.select(
    "CODMES",
    (F.col("N") - F.col("N_2")).alias("DIF_N"),
    F.round((F.col("MONTO") - F.col("MONTO_2")), 2).alias("DIF_MONTO"),
    F.round((F.col("PD_VIG") - F.col("PD_VIG_2")), 2).alias("DIF_PD_VIG"),
    F.round((F.col("RD24") - F.col("RD24_2")), 2).alias("DIF_RD24"),
    F.round((F.col("RD18") - F.col("RD18_2")), 2).alias("DIF_RD18"),
    F.round((F.col("DEF24") - F.col("DEF24_2")), 2).alias("DIF_DEF24")
)

# COMMAND ----------

# DBTITLE 1,Validacion:diferencia
# Mostrar el DataFrame resultante
cubo_diff.orderBy('CODMES').show(60)

# COMMAND ----------

# import inspect
# inspect.signature(Obj_Seg.EvolCal).parameters

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Afluente

# COMMAND ----------

Obj_Seg_cal, rc_1, rm_1, fc_1, fm_1 = Obj_Seg.EvolCal(
  # Configuración de filtro de uso y pd de calibración
  query_uso          = "CODMES>=202101 and segmento_banca_f='1. Afluente'",
  pd_seg             = pd_vig,
  pd_comparacion     = True,
  proys_def_temp     = 6,
  proys_mor_temp     = 0,
  rd_aux             = 'RD21',
  rd_aux2            = 'RD12', # Corregido
  mora_temp          = 'MORA_30_3',

  # Configuración de proyecciones por cuentas
  proy_def_temp_xc   = "np.where( (calxcuentas['CODMES'] == 202306) | (calxcuentas['CODMES'] == 202307) ,0.005, met1_xc)", #All_proy met1_xc met2_xc met3_xc
  suav_def_temp_xc   = 1,
  proy_mor_temp_xc   = 'det1_xc',
  suav_mor_temp_xc   = 1,
  prof_hist_xc       = 14,

  # Configuración de proyecciones por montos
  proy_def_temp_xm   = "np.where( (calxmontos['CODMES'] == 202306) | (calxmontos['CODMES'] == 202307) ,0.005, met1_xm)", #All_proy met1_xm met2_xm met3_xm
  suav_def_temp_xm   = 1,
  proy_mor_temp_xm   = 'det1_xm',
  suav_mor_temp_xm   = 1,
  prof_hist_xm       = 14,
  #fact_to_proy_xc    = [fc_af, fc_con],
  #fact_to_proy_xm    = [fm_af, fm_con],

  # Detalles del gráfico de calibración
  titulo             = titulo,
  pos_leyenda        = (0.5, -0.25),
  eje_y_xc           = 0.05,
  eje_y_xm           = 0.05,
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

fm_1

# COMMAND ----------

rm_1

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Consumo

# COMMAND ----------

Obj_Seg_cal, rc_2, rm_2, fc_2, fm_2 = Obj_Seg.EvolCal(
  # Configuración de filtro de uso y pd de calibración
  query_uso          = "CODMES>=202101 and segmento_banca_f='2. Consumo'",
  pd_seg             = pd_vig,
  pd_comparacion     = True,
  proys_def_temp     = 6,
  proys_mor_temp     = 0,
  rd_aux             = 'RD21',
  rd_aux2            = 'RD12', # Corregido
  mora_temp          = 'MORA_30_3',

  # Configuración de proyecciones por cuentas
  proy_def_temp_xc   =  "np.where( (calxcuentas['CODMES'] == 202304) | (calxcuentas['CODMES'] == 202303) | ( calxcuentas['CODMES'] == 202307),0.005, met1_xc) ", #All_proy met1_xc met2_xc met3_xc
  suav_def_temp_xc   = 1,
  proy_mor_temp_xc   = 'det1_xc',
  suav_mor_temp_xc   = 1,
  prof_hist_xc       = 14,

  # Configuración de proyecciones por montos
  proy_def_temp_xm   = "np.where( (calxmontos['CODMES'] == 202304) | (calxmontos['CODMES'] == 202303) | ( calxmontos['CODMES'] == 202307),0.005, met1_xm) ", #All_proy met1_xm met2_xm met3_xm
  suav_def_temp_xm   = 1,
  proy_mor_temp_xm   = "det1_xc",
  suav_mor_temp_xm   = 1,
  prof_hist_xm       = 14,
  #fact_to_proy_xc    = [fc_af, fc_con],
  #fact_to_proy_xm    = [fm_af, fm_con],

  # Detalles del gráfico de calibración
  titulo             = titulo,
  pos_leyenda        = (0.5, -0.25),
  eje_y_xc           = 0.2,
  eje_y_xm           = 0.2,
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

fm_2

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Total

# COMMAND ----------

Obj_Seg_cal, rc, rm, fc, fm = Obj_Seg.EvolCal(
  # Configuración de filtro de uso y pd de calibración
  query_uso          = "CODMES>=202101",
  pd_seg             = pd_vig,
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
  fact_to_proy_xc    = [fc_1, fc_2],
  fact_to_proy_xm    = [fm_1, fm_2],

  # Detalles del gráfico de calibración
  titulo             = titulo,
  pos_leyenda        = (0.5, -0.25),
  eje_y_xc           = 0.05,
  eje_y_xm           = 0.05,
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

#rm

# COMMAND ----------

periodo_proy_1=202303
periodo_proy_6=202308

# COMMAND ----------

#PD y RD proy
rm['PD_APP_HIP_TRDPYME_2Q24'][(rm['CODMES']>=periodo_proy_1) & (rm['CODMES']<=periodo_proy_6)].mean(), rm['RD_PROY_DEF'][(rm['CODMES']>=periodo_proy_1) & (rm['CODMES']<=periodo_proy_6)].mean()

# COMMAND ----------

rc['PD_APP_HIP_TRDPYME_2Q24'][(rc['CODMES']>=periodo_proy_1) & (rc['CODMES']<=periodo_proy_6)].mean(), rc['RD_PROY_DEF'][(rc['CODMES']>=periodo_proy_1) & (rc['CODMES']<=periodo_proy_6)].mean()

# COMMAND ----------

driver = 'segmento_banca_f'
Obj_SegPSI, mix_n, mix_m, psi_c, psi_m = Obj_Seg.MixPSI(
  # Selección del driver y filtro de uso
  driver             = driver,
  query_uso          = 'CODMES >= 202101',
  cast_int           = False,

  # Ventana de construcción o más antigua
  codmes_inicio1     = 202101,
  codmes_fin1        = 202312,

  # Periodo reciente
  codmes_inicio2     = 202401,
  codmes_fin2        = 202412,

  # Detalles del gráfico
  titulo             = driver,
  dim_grafico        = (22.5, 6),
  pos_leyenda        = (0.5, -0.25)
)

# COMMAND ----------

driver = 'SEGMENTO_ESPJ'
Obj_SegPSI, mix_n, mix_m, psi_c, psi_m = Obj_Seg.MixPSI(
  # Selección del driver y filtro de uso
  driver             = driver,
  query_uso          = 'CODMES >= 202101',
  cast_int           = False,

  # Ventana de construcción o más antigua
  codmes_inicio1     = 202101,
  codmes_fin1        = 202204,

  # Periodo reciente
  codmes_inicio2     = 202301,
  codmes_fin2        = 202404,

  # Detalles del gráfico
  titulo             = driver,
  dim_grafico        = (22.5, 6),
  pos_leyenda        = (0.5, -0.25)
)

# COMMAND ----------

VARIABLE = 'SEGMENTO_ESPJ'
caldri = Obj_Seg.EvolCalDri(
  variable = VARIABLE,
  # Configuración de filtro de uso y pd de calibración
  query_uso          = f"CODMES >= 202201",
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
  tipo_vista         = 'C',
  ncolumnas          = 3,
  nfilas             = 2,
  dim_grafico_total  = (30, 12),
  ncol_leyenda       = 7,
  tamaño_leyenda     = 8.5,
  etiquetas_total    = True,
  pos_etiqueta       = 0,
  vspacio            = -0.5,
  hspacio            = 0.5
)


# COMMAND ----------

Obj_Seg_cal, rc, rm, fc, fm = Obj_Seg.EvolCal(
  # Configuración de filtro de uso y pd de calibración
  query_uso          = "CODMES>=202101 and SEGMENTO_ESPJ = 'A'",
  pd_seg             = pd_vig,
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
  fact_to_proy_xc    = [fc_1, fc_2],
  fact_to_proy_xm    = [fm_1, fm_2],

  # Detalles del gráfico de calibración
  titulo             = titulo,
  pos_leyenda        = (0.5, -0.25),
  eje_y_xc           = 0.05,
  eje_y_xm           = 0.05,
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
# MAGIC ##### 1.2. Discriminación

# COMMAND ----------

import inspect
inspect.signature(Obj_Seg.EvolDis).parameters

# COMMAND ----------

Obj_SegDis, evoldis_gini, evoldis_ks = Obj_Seg.EvolDis(
  query_filtro = "CODMES>=201901 AND DEF_120_BLOQ_REF_0 = 0",
  pd_dis       = pd_vig,
  codmes_ini   = 202110,
  codmes_fin   = 202312,
  amplitud     = 6,
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

Obj_SegDis, evoldis_gini, evoldis_ks = Obj_Seg.EvolDis(
  query_filtro = "CODMES>=201901 AND DEF_120_BLOQ_REF_0 = 0",
  pd_dis       = pd_rbm,
  codmes_ini   = 202110,
  codmes_fin   = 202312,
  amplitud     = 6,
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

Obj_SegDis, evoldis_gini, evoldis_ks = Obj_Seg.EvolDis(
  query_filtro = "CODMES>=201901 AND DEF_120_BLOQ_REF_0 = 0",
  pd_dis       = pd_cal,
  codmes_ini   = 202110,
  codmes_fin   = 202312,
  amplitud     = 6,
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

evoldis_gini

# COMMAND ----------

evoldis_gini['GINI24'].mean()

# COMMAND ----------

Obj_SegDis2, evoldis_vig_gini, evoldis_vig_ks = Obj_Seg.EvolDis(
  query_filtro = "CODMES>=201901 AND DEF_120_BLOQ_REF_0 = 0",
  pd_dis       = pd_vig,
  codmes_ini   = 201906,
  codmes_fin   = 202312,
  amplitud     = 6,
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

evoldis_vig_gini

# COMMAND ----------

evoldis_vig_gini['GINI24'].mean()

# COMMAND ----------

Obj_SegDis3, evoldis_trc_gini, evoldis_trc_ks = Obj_Seg.EvolDis(
  query_filtro = "CODMES>=201901 AND DEF_120_BLOQ_REF_0 = 0",
  pd_dis       = pd_trc,
  codmes_ini   = 202110,
  codmes_fin   = 202312,
  amplitud     = 6,
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

# Definir las listas de variables
# Variables Negocio
var_neg = ["MONTO_R", "RAN_PLAZO", "PRS_TASA_Q", "SEGMENTO_BANCA_F", "TIPO_EMPLEO_F", "EDAD_R"]
var_neg_st = ["MTOAPROBADO_SOLES", "CTDPLAZOAPROBADO", "PRS_TASA", "SEGMENTO_BANCA_F", "TIPO_EMPLEO_F", "EDAD"]

# Variables Generales
var_gen = ["PAUTA", "LTV_CORREGIDO_0_Q", "INGRESO_SOL_CONY_Q", "END_TOT_SF_CY_Q", "FLG_PDH_2_3", "INGRESO_RBM_Q", "END_TOT_SF_Q", "MARCA_HML_T", "INGRESO_CONY_TIT_Q"]
var_gen_st=["PAUTA", "LTV_CORREGIDO_0", "INGRESO_SOL_CONY", "END_TOT_SF_CY", "FLG_PDH_2_3", "INGRESO_RBM", "END_TOT_SF", "MARCA_HML_T", "INGRESO_CONY_TIT"]

# Variables Calibrados
var_cal = ["DEU_NOHIP_6_24_Q", "UTIL_PROM_6_24_Q", "PAS_MIN_PROM24_Q", "ZONA_PAUTA5_F", "FLG_NODEP_ESPJ", "FLG_SOLTERO_ESPJ"]#, "SEGMENTO_ESPJ"
var_cal_st = ["EXP_PCT_EVOL_SHIP_U6M_RT_U24_A3", "RCC_PCT_UTL6_UTL24_RT_U24_A2", "SLD_PRM_PAS_MIN_24_24_RT_U24_I", "ZONA_PAUTA5_st", "FLG_NODEP_ESPJ", "FLG_SOLTERO_ESPJ"]#, "SEGMENTO2_ST"

# Variables modulos/troncales
xb_mod = ["XB_RCC_S_ADCW", "XB_PAS_S_ADCW", "XB_HIP", "XB_ACT_S_ADCW", "XB_DEM_S_ADCW", "XB_TRX_S_ADCW"]
var_mod = ["LTV_1_Q", "RIESGO_ZONA_ST", "GRUPO_PLAZO", "ANT_LABORAL_MESES_Q", "CAT_ZONA", "CIIU_DEF_Q", "EDAD_Q", "EST_CIV", "ING_CASCADA3_Q"]
var_mod_st = ["LTV_1_D2","RIESGO_ZONA_ST_ST","GRUPO_PLAZO_ST", "ANT_LABORAL_MESES_E2", "CAT_ZONA1_C_ST", "CL_CIIU_ST", "EDAD_E1", "EST_CIV_ST", "ING_CASCADA3_D2"]

# COMMAND ----------

# DBTITLE 1,nueva fuente
# Variables modulos/troncales nueva fuente
var_mod_ant = ["LTV_1_Q", "RIESGO_ZONA_ST", "GRUPO_PLAZO","ANT_LABORAL_MESES_Q" , "CAT_ZONA","EDAD_Q", "EST_CIV","INGRESO_SOL_CONY_Q", "INGRESO_RBM_Q"]
var_mod_nue = ["LTV_CORREGIDO_VF_Q", "NFT_RIESGO_ZONA_ST", "NFT_PLAZO_Q","SUITE_AntLaboralMeses_Q", "SUITE_CAT_ZONA", "SUITE_Edad_Q","SUITE_Estcivil","NFT_IngresoConyuge_Q","NFT_IngresoTitular_Q"]

# COMMAND ----------

Obj_Seg.base_monitoreo = Obj_Seg.base_monitoreo.withColumn("LTV_CORREGIDO_VF_Q",
                       when(col("LTV_CORREGIDO_VF").between(0.679, 0.801), '3. 0.679 - 0.801')
                       .when(col("LTV_CORREGIDO_VF") == 0.643, '1. 0.643 - 0.643')
                       .when(col("LTV_CORREGIDO_VF").between(0.643, 0.679), '2. 0.643 - 0.679')
                       .when(col("LTV_CORREGIDO_VF").between(0.801, 0.888), '4. 0.801 - 0.888')
                       .when(col("LTV_CORREGIDO_VF").isNull(), '98. Missing')
                       .otherwise('99. Out of Range'))

# COMMAND ----------

Obj_Seg.base_monitoreo  = Obj_Seg.base_monitoreo .withColumn("NFT_PLAZO_Q",
                       when(col("NFT_PLAZO").between(10, 14), '3. De 10 a 14 años')
                       .when(col("NFT_PLAZO").between(5, 9), '2. De 5 a 9 años')
                       .when(col("NFT_PLAZO").between(25, 29), '6. De 25 a 29 años')
                       .when(col("NFT_PLAZO") >= 35, '8. 35 años o más')
                       .when(col("NFT_PLAZO").between(15, 19), '4. De 15 a 19 años')
                       .when(col("NFT_PLAZO").between(20, 24), '5. De 20 a 24 años')
                       .when(col("NFT_PLAZO") < 5, '1. Menor a 4 años')
                       .when(col("NFT_PLAZO").isNull(), '98. Missing')
                       .otherwise('99. Out of Range'))

# COMMAND ----------

Obj_Seg.base_monitoreo = Obj_Seg.base_monitoreo.withColumn("SUITE_AntLaboralMeses_Q",
                       when(col("SUITE_AntLaboralMeses").between(-41.000, 27.000), '1. -41.000 - 27.000')                   
                       .when(col("SUITE_AntLaboralMeses").between(27.000, 59.000), '2. 27.000 - 59.000')
                       .when(col("SUITE_AntLaboralMeses").between(59.000, 114.000), '3. 59.000 - 114.000')
                       .when(col("SUITE_AntLaboralMeses").between(114.000, 610.000), '4. 114.000 - 610.000')
                       .when(col("SUITE_AntLaboralMeses").isNull(), '98. Missing')
                       .otherwise('99. Out of Range'))

# COMMAND ----------

Obj_Seg.base_monitoreo = Obj_Seg.base_monitoreo.withColumn("SUITE_Edad_Q",
                       when(col("SUITE_Edad").between(20.000, 34.000), '1. 20.000 - 34.000')
                       .when(col("SUITE_Edad").between(34.000, 39.000), '2. 34.000 - 39.000')
                       .when(col("SUITE_Edad").between(39.000, 45.000), '3. 39.000 - 45.000')
                       .when(col("SUITE_Edad").between(45.000, 70.000), '4. 45.000 - 70.000')
                       .when(col("SUITE_Edad").isNull(), '98. Missing')
                       .otherwise('99. Out of Range'))

# COMMAND ----------

Obj_Seg.base_monitoreo  = Obj_Seg.base_monitoreo .withColumn("NFT_IngresoConyuge_Q",
                        when(col("NFT_IngresoConyuge")==0 ,'0. Sin conyuge')
                       .when(col("NFT_IngresoConyuge").between(0.000, 7929.530), '1. 0.000 - 7929.530')
                       .when(col("NFT_IngresoConyuge").between(7929.530, 11225.000), '2. 7929.530 - 112...')
                       .when(col("NFT_IngresoConyuge").between(11225.000, 17434.000), '3. 11225.000 - 17...')
                       .when(col("NFT_IngresoConyuge").between(17434.000, float('inf')), '4. 17434.000 - 82...')
                       .when(col("NFT_IngresoConyuge").isNull(), '98. Missing')
                       .otherwise('99. Out of Range'))

# COMMAND ----------

Obj_Seg.base_monitoreo = Obj_Seg.base_monitoreo.withColumn("NFT_IngresoTitular_Q",
                       when(col("NFT_IngresoTitular").between(0.000, 6159.000), '1. 0.000 - 6159.000')
                       .when(col("NFT_IngresoTitular").between(6159.000, 9409.000), '2. 6159.000 - 940...')
                       .when(col("NFT_IngresoTitular").between(9409.000, 16000.000), '3. 9409.000 - 160...')
                       .when(col("NFT_IngresoTitular").between(16000.000, float('inf')), '4. 16000.000 - 43...')
                       .when(col("NFT_IngresoTitular").isNull(), '98. Missing')
                       .otherwise('99. Out of Range'))

# COMMAND ----------

Obj_Seg.base_monitoreo = Obj_Seg.base_monitoreo.withColumn(
    "NFT_RIESGO_ZONA_ST",
    when(
        (col("NFT_departamentoinmueble") == 'LIMA') & 
        (~col("NFT_distritoinmueble").isin(
            'ATE-LIMA', 'CARABAYLLO-LIMA', 'EL AGUSTINO-LIMA', 
            'LURIN-LIMA', 'PACHACAMAC-LIMA', 'PUENTE PIEDRA-LIMA', 
            'SAN JUAN LURIGANCHO', 'SAN JUAN MIRAFLORES', 
            'VILLA EL SALVADOR', 'VILLA MARIA TRIUNFO'
        )), 'BAJO'
    ).when(
        (col("NFT_departamentoinmueble") == 'CALLAO') & 
        (col("NFT_distritoinmueble") != 'VENTANILLA-LIMA'), 'BAJO'
    ).otherwise('ALTO')
)


# COMMAND ----------


# Agregar la columna NFT_RIESGO_ZONA_ST con las condiciones especificadas
Obj_Seg.base_monitoreo = Obj_Seg.base_monitoreo.withColumn("NFT_RIESGO_ZONA_ST", when((col("NFT_departamentoinmueble") == "CALLAO") & (col("NFT_distritoinmueble") == "VENTANILLA-LIMA"), "ALTO")
      .when((col("NFT_departamentoinmueble") == "LIMA") & (col("NFT_distritoinmueble").isin(["ATE-LIMA", "CARABAYLLO-LIMA", "EL AGUSTINO-LIMA", 
                                                                             "LURIN-LIMA", "PACHACAMAC-LIMA", "PUENTE PIEDRA-LIMA",
                                                                             "SAN JUAN LURIGANCHO", "SAN JUAN MIRAFLORES",
                                                                             "VILLA EL SALVADOR", "VILLA MARIA TRIUNFO"])), "ALTO")
      .when((col("NFT_departamentoinmueble").isNull()) | (col("NFT_departamentoinmueble") == "") | (col("NFT_distritoinmueble").isNull()) | (col("NFT_distritoinmueble") == ""), "ND")
      .otherwise("BAJO"))

# Agregar la columna NFT_RIESGO_ZONA_ST_ST con las condiciones especificadas
Obj_Seg.base_monitoreo = Obj_Seg.base_monitoreo.withColumn("NFT_RIESGO_ZONA_ST_ST", when(col("NFT_RIESGO_ZONA_ST") == "ALTO", -4.62497281328427)
      .when(col("NFT_RIESGO_ZONA_ST") == "BAJO", -5.21697172709408)
      .when(col("RIESGO_ZONA_ST") == "ND", -4.61470207761345))


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

Obj_Seg.base_monitoreo = Obj_Seg.base_monitoreo.withColumn("SUITE_CAT_ZONA", 
                   when(col("SUITE_desprovincia") == 'LIMA', 'LIMA')
                   .when(col("SUITE_desprovincia").isNull(), '99. Missing')
                   .otherwise('PROVINCIA'))

# COMMAND ----------

# MAGIC %md
# MAGIC - LTV_CORREGIDO_VF =(NFT_LTV+LTV_CORREGIDO_0) =>LTV corregido de nueva fuente desde ago23
# MAGIC - LTV_CORREGIDO_0 = (LTV_EXT + LTV_RBM_ORACLE) => LTV con info mal desde jul23
# MAGIC - CSOL_LTV = solicitudes de fuente solicitud de creditos hipoteario 
# MAGIC - CSOL_SC_RBM = score rbm 
# MAGIC - NFT_LTV = LTV con la futura info productiva
# MAGIC - PRS_TASA = tasa de hipotecario con fuente prestamo

# COMMAND ----------

columnas = ["PRS_TASA", "LTV_CORREGIDO_0",  "END_TOT_SF_CY","END_TOT_SF", "DEU_NOHIP_6_24", "UTIL_PROM_6_24", "PAS_MIN_PROM24"]
condicion_filtro ="CODMES BETWEEN 202101 AND 202410 AND DEF_120_BLOQ_REF_0 = 0"

# Aplicar la función de categorización
Obj_Seg.base_monitoreo = categorize_by_quintiles(Obj_Seg.base_monitoreo, columnas, condicion_filtro)

# COMMAND ----------

# Variables a categorizar
columnas = ["LTV_1", "ANT_LABORAL_MESES", "EDAD", "INGRESO_SOL_CONY","INGRESO_RBM"]
condicion_filtro = "CODMES  between  202101 and 202201"

# Aplicar la función de categorización
Obj_Seg.base_monitoreo = categorize_by_quintiles(Obj_Seg.base_monitoreo, columnas, condicion_filtro)

# COMMAND ----------

# Variables a categoricas
columnas = ["PD_APP_HIP_TRDPYME_2Q24","PD_APP_TRAD","ING_CASCADA3","CIIU_DEF"]
condicion_filtro = "CODMES BETWEEN 202101 AND 202410 AND DEF_120_BLOQ_REF_0 = 0"
Obj_Seg.base_monitoreo = categorize_by_quintiles(Obj_Seg.base_monitoreo, columnas, condicion_filtro)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2.1 Calibracion Driver

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 2.1.1. Drivers negocio

# COMMAND ----------

query_cal="CODMES BETWEEN 202209 AND 202302"

# COMMAND ----------


caldri = Obj_Seg.CalVar(
  # Parametros de figura individual
  query_uso          = query_cal,
  pd_seg             = pd_vig,
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

caldri = Obj_Seg.CalVar(
  # Parametros de figura individual
  query_uso          = query_cal,
  pd_seg             = pd_vig,
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
    tipo_vista = 'M',
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
    amplitud     = 6,
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

Obj_Seg_cal, rc_bck, rm_bck, fc, fm = Obj_Seg.EvolCal(
  # Configuración de filtro de uso y pd de calibración
  query_uso          = "CODMES>=202101 and MTOAPROBADO_SOLES>=680000",
  pd_seg             = pd_vig,
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
  fact_to_proy_xc    = [fc_1, fc_2],
  fact_to_proy_xm    = [fm_1, fm_2],

  # Detalles del gráfico de calibración
  titulo             = titulo,
  pos_leyenda        = (0.5, -0.25),
  eje_y_xc           = 0.1,
  eje_y_xm           = 0.1,
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

Obj_Seg_cal, rc_bck2, rm_bck2, fc, fm = Obj_Seg.EvolCal(
  # Configuración de filtro de uso y pd de calibración
  query_uso          = "CODMES>=202101 and MTOAPROBADO_SOLES between 460000 and 680000",
  pd_seg             = pd_vig,
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
  fact_to_proy_xc    = [fc_1, fc_2],
  fact_to_proy_xm    = [fm_1, fm_2],

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
  exportar_factores  = True,
  mat_porc=-24
)

# COMMAND ----------

Obj_Seg_cal, rc_bck3, rm_bck3, fc, fm = Obj_Seg.EvolCal(
  # Configuración de filtro de uso y pd de calibración
  query_uso          = "CODMES>=202101 and MTOAPROBADO_SOLES between 340000 and 460000",
  pd_seg             = pd_vig,
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
  fact_to_proy_xc    = [fc_1, fc_2],
  fact_to_proy_xm    = [fm_1, fm_2],

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
  exportar_factores  = True,
  mat_porc=-24
)

# COMMAND ----------

Obj_Seg_cal, rc_bck4, rm_bck4, fc, fm = Obj_Seg.EvolCal(
  # Configuración de filtro de uso y pd de calibración
  query_uso          = "CODMES>=202101 and MTOAPROBADO_SOLES between 237500 and 340000",
  pd_seg             = pd_vig,
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
  fact_to_proy_xc    = [fc_1, fc_2],
  fact_to_proy_xm    = [fm_1, fm_2],

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
  exportar_factores  = True,
  mat_porc=-24
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 2.1.2. Drivers generales

# COMMAND ----------


caldri = Obj_Seg.CalVar(
  # Parametros de figura individual
  query_uso          = query_cal,
  pd_seg             = pd_vig,
  rd_aux             = 'RD12',
  pd_comparacion     = True, #Para tomar mas de un PD
  factores_cuentas   = fc,
  factores_montos    = fm,
  mat_importancia    = 5, # minimo % de materialidad por bucket mostrado
  cast_int           = True, #para los flag los cambia a entero
  dim_grafico        = (25, 6),
  pos_leyenda        = (0.5, -0.35),
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
  vspacio           = -0.25,
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
    codmes_ini   = 202110,
    codmes_fin   = 202312,
    amplitud     = 6,
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
  query_uso          = query_cal,
  pd_seg             = pd_vig,
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
    codmes_ini   = 202110,
    codmes_fin   = 202312,
    amplitud     = 6,
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

Obj_Seg_Caldri, rc_dri, rm_dri = Obj_Seg.CalDri(
  driver             = 'PD_APP_TRAD_Q',
  query_uso          = f"CODMES BETWEEN 202209 AND 202302 AND PD_APP_TRAD_Q NOT LIKE '%ssing%'",
  pd_seg             = pd_vig,
  rd_aux             = 'RD6',
  pd_comparacion     = True,
  factores_cuentas   = fc,
  factores_montos    = fm,
  mat_importancia    = 0,
  cast_int           = True, 
  titulo             = f'PD_APP_TRAD',
  dim_grafico        = (25, 6),
  etiquetas          = True,
  pos_etiquetas_xc   = [10, 10, 10, 10, 10],
  pos_etiquetas_xm   = [10, 10, 10, 10, 10],
  pos_leyenda        = (0.5, -0.2),
  punt_mora          = 100
) 

# COMMAND ----------

Obj_Seg_Caldri, rc_dri, rm_dri = Obj_Seg.CalDri(
  driver             = 'PD_APP_HIP_TRDPYME_2Q24_Q',
  query_uso          = f"CODMES BETWEEN 202209 AND 202302 AND PD_APP_HIP_TRDPYME_2Q24_Q NOT LIKE '%ssing%'",
  pd_seg             = pd_vig,
  rd_aux             = 'RD6',
  pd_comparacion     = True,
  factores_cuentas   = fc,
  factores_montos    = fm,
  mat_importancia    = 0,
  cast_int           = True, 
  titulo             = f'PD_APP_HIP_TRDPYME_2Q24',
  dim_grafico        = (25, 6),
  etiquetas          = True,
  pos_etiquetas_xc   = [10, 10, 10, 10, 10],
  pos_etiquetas_xm   = [10, 10, 10, 10, 10],
  pos_leyenda        = (0.5, -0.2),
  punt_mora          = 100
) 

# COMMAND ----------

# MAGIC %md
# MAGIC ####3. Vista Troncales

# COMMAND ----------

gini = Obj_Seg.EvolDisVar(
  # Parametros de figura individual
    query_filtro = '',
    codmes_ini   = 202110,
    codmes_fin   = 202312,
    amplitud     = 6,
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


caldri = Obj_Seg.CalVar(
  # Parametros de figura individual
  query_uso          = f"CODMES BETWEEN 202209 AND 202302",
  pd_seg             = pd_vig,
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
  nfilas    = 3,
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
    dim_grafico=(22.5, 6),
    pos_leyenda=(0.5, -0.35),
 
    # Parametros de figura final
    tipo_vista = 'M',
    ncolumnas = 3,
    nfilas    = 3,
    variables = var_mod,
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
    codmes_ini   = 202110,
    codmes_fin   = 202312,
    amplitud     = 6,
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
    variables = var_mod_st,
    dim_grafico_total = (25, 15),
    ncol_leyenda = 5,
    vspacio=-0.25,
    hspacio=4
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
    variables = var_mod_ant,
    dim_grafico_total = (25, 15),
    ncol_leyenda=8,
    tamaño_leyenda = 8.25,
    vspacio=-0.25,
    hspacio=0
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
    variables = var_mod_nue,
    dim_grafico_total = (25, 15),
    ncol_leyenda=8,
    tamaño_leyenda = 8.25,
    vspacio=-0.25,
    hspacio=0
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

df_factores=procesar_factores(2,202502)

# COMMAND ----------

df_factores

# COMMAND ----------

df_factores = spark.createDataFrame(df_factores)
df_factores.createOrReplaceTempView("temp_view")
spark.sql("CREATE OR REPLACE TABLE catalog_lhcl_prod_bcp.bcp_edv_fabseg.T45988_FAC_ADM_HIP_TRAD_202503 LOCATION 'abfss://bcp-edv-fabseg@adlscu1lhclbackp05.dfs.core.windows.net/HIPOTECARIO_2025/APP/202502/T45988_FAC_ADM_HIP_TRAD_202503' AS SELECT * FROM temp_view")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.2. Vista Cubo

# COMMAND ----------

pd_vig='PD_APP_TRAD' 
pd_cal='PD_APP_HIP_TRDPYME_2Q24'
pd_trc='XB_ADC_POND85_F'
tipo_banda = 'Jeffrey'#Jeffrey Vasiseck
codmes_default = 202302
columna_monto='MTOAPROBADO_SOLES'
titulo='Calibración Applicant App Tradicional'

# COMMAND ----------

#from pyspark.sql import SparkSession
#base_seguimiento = spark.sql("select distinct * from catalog_lhcl_prod_bcp.bcp_edv_fabseg.T45988_HM_ADM_HIP_TRAD_202502_F")

# COMMAND ----------

#base_seguimiento = base_seguimiento.filter("CODMES>=202101 AND DEF_120_BLOQ_REF_0 = 0 AND PD_APP_HIP_TRDPYME_2Q24 IS NOT NULL")

# COMMAND ----------

Obj_Seg.base_monitoreo = Obj_Seg.base_monitoreo.withColumn("SEG_PROYECCION", when(Obj_Seg.base_monitoreo.SEGMENTO_BANCA_F == "1. Afluente", 1).when(Obj_Seg.base_monitoreo.SEGMENTO_BANCA_F == "2. Consumo", 2))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.3. Formación cubo

# COMMAND ----------

# Definir la fecha base
MES_DEF_ACT = 202502
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

monto = "MTOAPROBADO_SOLES"

# COMMAND ----------

#CÁLCULO DE VALORES DE MESES Y CRUCE

from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Definir la fecha base
MES_DEF_ACT = 202502
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

# Cargar las tablas para el cruce
df_a = Obj_Seg.base_monitoreo
df_b = spark.table("catalog_lhcl_prod_bcp.bcp_edv_fabseg.T45988_FAC_ADM_HIP_TRAD_202503")

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

#-------------------------------------------------------------------------------------------------------------------

#CUBO PRINCIPAL

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when, count

# Crear las columnas flgventana_diagnostico, flgventana_reciente, flgdef_cerrado, flgdef_temp, flgmora_temp
df_cal01 = df_cal01.withColumn("flgventana_diagnostico", when(col("CODMES").between(codmes_a11, codmes_a6), 1).otherwise(0)) \
           .withColumn("flgventana_reciente", when(col("CODMES").between(202410, 202412), 1).otherwise(0)) \
           .withColumn("flgdef_cerrado", when(col("CODMES") <= codmes_a12, 1).otherwise(0)) \
           .withColumn("flgdef_temp", when(col("CODMES") <= codmes_a9, 1).otherwise(0)) \
           .withColumn("flgmora_temp", when(col("CODMES") <= 202401, 1).otherwise(0)) \
           .withColumn("PASA_PAUTA", when(col("SC_APP_TRAD") >= 335, 1).otherwise(0))
           
#Creación de nuevas PDs
# Calcular XB_TRAD a partir de CSOL_SC_RBM (SC_TRAD)
df_cal01 = df_cal01.withColumn('XB_TRAD', expr("(174.25 - CSOL_SC_RBM) / 57.708"))

# Calcular PD_TRAD a partir de XB_TRAD y renombrar la columna resultante a PD_RBM
df_cal01 = df_cal01.withColumn('PD_RBM', expr("1 / (1 + exp(-XB_TRAD))"))

# Calcular PD_TRAD para las columnas XB especificadas y renombrar las columnas resultantes
xb_columns = {
    'XB_ADC_POND85_F': 'PD_TRC',
    'XB_HIP': 'PD_HIP',
    'XB_DEM_S_ADCW': 'PD_DEMOGRAFICO',
    'XB_ACT_S_ADCW': 'PD_ACTIVO',
    'XB_TRX_S_ADCW': 'PD_TRANSACCIONAL',
    'XB_PAS_S_ADCW': 'PD_PASIVO',
    'XB_RCC_S_ADCW': 'PD_RCC'
}

for col_name, new_col_name in xb_columns.items():
    df_cal01 = df_cal01.withColumn(new_col_name, expr(f"1 / (1 + exp(-{col_name}))"))

# Eliminar la columna temporal XB_TRAD
df_cal01 = df_cal01.drop('XB_TRAD')

# LISTA DE NOMBRES DE COLUMNAS
lista_pd = [
    "PD_APP_TRAD",
    "PD_APP_HIP_TRDPYME_2Q24",
    "PD_TRC",
    "PD_RBM",
    "PD_HIP",
    "PD_DEMOGRAFICO",
    "PD_ACTIVO",
    "PD_TRANSACCIONAL",
    "PD_PASIVO",
    "PD_RCC"
]
# Crear una lista de expresiones de agregación
pd_result_xm = [sum(col(c) * col(monto)).alias(f"pd_{str(i+1).zfill(2)}_xm") for i, c in enumerate(lista_pd)]
# Crear una lista de expresiones de agregación sin multiplicar por monto
pd_result_xn = [sum(col(c)).alias(f"pd_{str(i+1).zfill(2)}_xn") for i, c in enumerate(lista_pd)]

#DEFINIR LISTA DE EJES 
lista_ejes = {
    "CAMPANIA_AGRUPADA": "EJE_P1",
    "TIPO_FONDEO": "EJE_P2",
    "SEGMENTO_BANCA_F": "EJE_P3",  
    "FLG_BANCARIZADO": "EJE_P4",
    "RANGO_SCORE": "EJE_P5",
    "DEU_NOHIP_6_24_Q": "EJE_S1",
    "UTIL_PROM_6_24_Q": "EJE_S2",
    "PAS_MIN_PROM24_Q": "EJE_S3",
    "ZONA_PAUTA5_F": "EJE_S4",
    "FLG_NODEP_ESPJ": "EJE_S5",
    "FLG_SOLTERO_ESPJ": "EJE_S6",
    "MONTO_R": "EJE_S7",
    "RAN_PLAZO": "EJE_S8",
    "PRS_TASA_Q": "EJE_S9",
    "TIPO_EMPLEO_F": "EJE_S10",
    "EDAD_R": "EJE_S11",
    "RAN_ING_CY": "EJE_S12",
    "LTV_CORREGIDO_0_Q": "EJE_S13",
    "INGRESO_RBM_Q": "EJE_S14",
    "MARCA_HML_T": "EJE_S15",
    "PASA_PAUTA": "EJE_S16"
}

# Crear una lista de expresiones de selección usando el diccionario
eje_result = [col(k).alias(v) for k, v in lista_ejes.items()]

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

#-------------------------------------------------------------------------------------------------------------------

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
    ("EJE_P4", "Flg Bancarizado"),
    ("EJE_P5", "Rango de Score"),
    ("EJE_S1", "Deuda promedio No Hipotecaria entre 6 y 24 m"),
    ("EJE_S2", "Utilizacion promedio de lineas entre 6 y 24 m"),
    ("EJE_S3", "Ratio entre el pasivo minimo y prom. en u24 m"),
    ("EJE_S4", "Zona Pauta"),
    ("EJE_S5", "Flag No Dependiente"),
    ("EJE_S6", "Flag Soltero"),
    ("EJE_S7", "Monto aprobado"),
    ("EJE_S8", "Rango Plazo"),
    ("EJE_S9", "Rango de Tasa"),
    ("EJE_S10", "Tipo de Empleo"),
    ("EJE_S11", "Rango de Edad"),
    ("EJE_S12", "Rango Ingreso Conyugue"),
    ("EJE_S13", "LTV"),
    ("EJE_S14", "Rango de Ingreso Titular"),
    ("EJE_S15", "Marca HML"),
    ("EJE_S16", "Pasa Pauta")
]

# Crear el DataFrame
df_etiquetas_ejes_final = spark.createDataFrame(data, schema)

df_etiquetas_ejes = df_etiquetas_ejes_final.toPandas()

#-------------------------------------------------------------------------------------------------------------------

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
    ("04", "04-PD RBM"),
    ("05", "05-PD Hip"),
    ("06", "06-PD Demográfico"),
    ("07", "07-PD Activo"),
    ("08", "08-PD Transaccional"),
    ("09", "09-PD Pasivo"),
    ("10", "10-PD RCC"),
]

# Crear el DataFrame
df_descriptivo_pds_final = spark.createDataFrame(data, schema)
# Convertir a Pandas
df_descriptivo_pds = df_descriptivo_pds_final.toPandas()

# COMMAND ----------

# PASAR DE DATAFRAME A UNA TABLA DE DATABRICKS
cubo_principal.createOrReplaceTempView("temp_view")

spark.sql("""
CREATE OR REPLACE TABLE catalog_lhcl_prod_bcp.bcp_edv_fabseg.cubo_trad 
LOCATION 'abfss://bcp-edv-fabseg@adlscu1lhclbackp05.dfs.core.windows.net/HIPOTECARIO_2025/APP/202502/cubo_trad' 
AS SELECT * FROM temp_view
""")

# COMMAND ----------

# MAGIC %md
# MAGIC #####4.4. Exportar CSV

# COMMAND ----------

#EXPORTAR EN ARCHIVO CSV (PARA EL CUBO_PRINCIPAL SE ESTÁ EXPORTANDO EN ZIP (COMPRIMIDO))

import zipfile

# Guardar los DataFrames como archivos CSV
df_etiquetas_ejes.to_csv("/Workspace/Users/mariapaloma@bcp.com.pe/App_Tradicional/202503/etiquetas_ejes.csv", index=False)
df_descriptivo_pds.to_csv("/Workspace/Users/mariapaloma@bcp.com.pe/App_Tradicional/202503/descriptivo_pds.csv", index=False)

# Guardar y comprimir cubo_principal_final
csv_path = "/Workspace/Users/mariapaloma@bcp.com.pe/App_Tradicional/202503/cubo_principal.csv"
zip_path = "/Workspace/Users/mariapaloma@bcp.com.pe/App_Tradicional/202503/cubo_principal.zip"

cubo_principal_final.to_csv(csv_path, index=False)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(csv_path, arcname="cubo_principal.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. Vista 2x2

# COMMAND ----------

VARIABLE = 'TIPO_EMPLEO_F'
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



# COMMAND ----------

VARIABLE = 'EDAD_R'
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
  ncolumnas          = 3,
  nfilas             = 2,
  dim_grafico_total  = (22, 12),
  ncol_leyenda       = 7,
  tamaño_leyenda     = 8.5,
  etiquetas_total    = True,
  pos_etiqueta       = 0,
  vspacio            = -0.5,
  hspacio            = 0.5
)

# COMMAND ----------

VARIABLE = 'EDAD_R'
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
  tipo_vista         = 'C',
  ncolumnas          = 2,
  nfilas             = 3,
  dim_grafico_total  = (22, 12),
  ncol_leyenda       = 7,
  tamaño_leyenda     = 8.5,
  etiquetas_total    = True,
  pos_etiqueta       = 0,
  vspacio            = -0.5,
  hspacio            = 0.5
)
