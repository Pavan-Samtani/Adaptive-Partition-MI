import codecs

TABLE = r"""
\begin{table*}[!t]
\centering
\normalsize
\caption{Tabla con valores de los promedios de la información mútua calculada para 50 casos, utilizando el estimador de máxima verosimilitud ($\hat I_{ML}$), con partición adaptiva ($\hat I_{CI}$) y con partición no adaptiva ($\hat I_{NA}$), para distintos valores de correlación $r$ utilizados en una distribución normal bivariada. Además de los valores $I$ de la información mútua real para cada caso}
\label{tabla:gauss}
\begin{tabular}{ll|l|l|l|l|l|ll}
\cline{3-7}
 &  & \multicolumn{5}{c|}{Número de muestras} &  &  \\ \cline{3-7} \cline{9-9} 
 &  & 250 & 500 & 1000 & 2000 & 10000 & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{I} \\ \cline{1-7} \cline{9-9} 
\multicolumn{1}{l|}{\multirow{3}{*}{$r=0$}} & avg($\hat I_{ML}$) & %s & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} \\ \cline{2-7}
\multicolumn{1}{l|}{} & avg($\hat I_{CI}$) & %s & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{%s} \\ \cline{2-7}
\multicolumn{1}{l|}{} & avg($\hat I_{NA}$) & %s & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} \\ \cline{1-7} \cline{9-9} 
\multicolumn{1}{l|}{\multirow{3}{*}{$r=0.3$}} & avg($\hat I_{ML}$) & %s & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} \\ \cline{2-7}
\multicolumn{1}{l|}{} & avg($\hat I_{CI}$) & %s & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{%s} \\ \cline{2-7}
\multicolumn{1}{l|}{} & avg($\hat I_{NA}$) & %s & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} \\ \cline{1-7} \cline{9-9} 
\multicolumn{1}{l|}{\multirow{3}{*}{$r=0.6$}} & avg($\hat I_{ML}$) & %s & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} \\ \cline{2-7}
\multicolumn{1}{l|}{} & avg($\hat I_{CI}$) & %s & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{%s} \\ \cline{2-7}
\multicolumn{1}{l|}{} & avg($\hat I_{NA}$) & %s & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} \\ \cline{1-7} \cline{9-9} 
\multicolumn{1}{l|}{\multirow{3}{*}{$r=0.9$}} & avg($\hat I_{ML}$) & %s & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} \\ \cline{2-7}
\multicolumn{1}{l|}{} & avg($\hat I_{CI}$) & %s & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{%s} \\ \cline{2-7}
\multicolumn{1}{l|}{} & avg($\hat I_{NA}$) & %s & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} \\ \cline{1-7} \cline{9-9} 
\end{tabular}
\end{table*}
"""

def generate_table(results):
    format_list = []
    prec_format = "{:.4f}"
    for sample_dict, real_mi in results.values():
        i_ml, i_ad, i_nad = zip(*sample_dict.values())
        format_list.append("& ".join(map(prec_format.format, i_ml)))
        format_list.append("& ".join(map(prec_format.format, i_ad)))
        format_list.append(prec_format.format(real_mi))
        format_list.append("& ".join(map(prec_format.format, i_nad)))

    with codecs.open("table.tex", 'w', encoding='utf-8') as f:
        f.write(TABLE % tuple(format_list))
