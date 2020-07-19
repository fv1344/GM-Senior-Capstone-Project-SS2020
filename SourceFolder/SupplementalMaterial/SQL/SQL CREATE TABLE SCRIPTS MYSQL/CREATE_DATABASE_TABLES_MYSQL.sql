
/*SCRIPT TO GENERATE ALL THE BACK-END DATA TABLES FOR GM FINTECH APPLICATION */                                             */
/*SCRIPT IS WRITTEN FOR SQL FORMAT ACCEPTED BY MYSQL 8.X                     */

use gmfsp_db;

DROP TABLE IF EXISTS dbo_datedim;
CREATE TABLE dbo_datedim(
`date`                 date,
`year`                 int,
`month`                int,
qtr                    int,
weekend                int,
isholiday              int,
PRIMARY KEY (`date`)
)
;

DROP TABLE IF EXISTS dbo_instrumentmaster;  
CREATE TABLE dbo_instrumentmaster(
instrumentid            int,
instrumentname          varchar(50),
`type`                  varchar(50),
exchangename            varchar(50),
PRIMARY KEY (instrumentid)
)
;

/* MUST INSERT VALUES INTO INSTRUMENTMASTER TABLE IN ORDER FOR DATABASE TO WORK */
/* USE INSERT_INTO_INSTRUMENT_MASTER_MYSQL.sql FILE TO ADD SYMBOLS              */
/* USE REMOVE_FROM_INSTRUMENT_MASTER_MYSQL.sql FILE TO REMOVE SYMBOLS           */

INSERT INTO dbo_instrumentmaster
VALUES (1, 'GM', 'Equity', 'YAHOO'),
	   (2, 'PFE', 'Equity', 'YAHOO'),
	   (3, 'SPY', 'Equity', 'YAHOO'),
	   (4, 'XPH', 'Equity', 'YAHOO'),
	   (5, 'CARZ', 'Equity', 'YAHOO'),
       (6, '^TYX', 'Equity', 'YAHOO'),
	   (7, 'FCAU', 'Equity', 'YAHOO'),
	   (8, 'TM', 'Equity', 'YAHOO'),
	   (9, 'F', 'Equity', 'YAHOO'),
	   (10, 'HMC', 'Equity', 'YAHOO')
;

DROP TABLE IF EXISTS dbo_macroeconmaster;  
CREATE TABLE dbo_macroeconmaster(
macroeconcode		varchar(50),
macroeconname		varchar(50),
accesssourcekey		varchar(50),
accesssource		varchar(50),
datecreated			date,
activecode			varchar(10),
PRIMARY KEY (macroeconcode)
)
;

INSERT INTO dbo_macroeconmaster
VALUES ('GDP', 'GDP', 'FRED/NGDPPOT', 'Quandl', 0, 'A'),
	   ('UR', 'Unemployment Rate', 'USMISERY/INDEX', 'Quandl', 0, 'A'),
	   ('IR', 'Inflation Rate', 'USMISERY/INDEX', 'Quandl', 0, 'A'),
	   ('MI', 'Misery Index', 'USMISERY/INDEX', 'Quandl', 0, 'A'),
       ('TYX', '30 Year Bond Yield', '^TYX', 'Yahoo', 0, 'I'),
       ('COVI', 'Crude Oil ETF Volatility Index', 'OVXCLS', 'FRED', 0, 'A'),
       ('CPIUC', 'Consumer Price Index Urban Consumers', 'CPIAUCSL', 'FRED', 0, 'A'),
       ('FSI', 'Financial Stress Index', 'STLFSI', 'FRED', 0, 'A')
;

DROP TABLE IF EXISTS dbo_instrumentstatistics;
CREATE TABLE dbo_instrumentstatistics(        
`date`				date,
high				float,
low					float,
`open`				float,
`close`         	float,
volume          	float,
`adj close`     	float,
instrumentid    	int,
FOREIGN KEY (`date`) REFERENCES dbo_datedim(`date`),
FOREIGN KEY (instrumentid) REFERENCES dbo_instrumentmaster(instrumentid)
)
;

DROP TABLE IF EXISTS dbo_macroeconstatistics;
CREATE TABLE dbo_macroeconstatistics (
`date`				date,
statistics			int,
macroeconcode		varchar(50),
FOREIGN KEY (`date`) REFERENCES dbo_datedim(`date`),
FOREIGN KEY (macroeconcode) REFERENCES dbo_macroeconmaster(macroeconcode)
)
;

DROP TABLE IF EXISTS dbo_strategymaster;
CREATE TABLE dbo_strategymaster (          
strategycode        varchar(20),
strategyname        varchar(50),
PRIMARY KEY (strategycode) 
)
;

DROP TABLE IF EXISTS dbo_algorithmmaster;
CREATE TABLE dbo_algorithmmaster (          
algorithmcode         varchar(50),
algorithmname         varchar(50),
PRIMARY KEY (algorithmcode)
)
;

DROP TABLE IF EXISTS dbo_algorithmforecast;
CREATE TABLE dbo_algorithmforecast (          
forecastdate            date,
instrumentid            int,
forecastcloseprice      float,
algorithmcode           varchar(50),
prederror               float,
FOREIGN KEY (forecastdate) REFERENCES dbo_datedim(`date`),
FOREIGN KEY (instrumentid) REFERENCES dbo_instrumentmaster(instrumentid),
FOREIGN KEY (algorithmcode) REFERENCES dbo_algorithmmaster(algorithmcode)
)
;

DROP TABLE IF EXISTS dbo_actionsignals;
CREATE TABLE dbo_actionsignals (          
`date`                date,
instrumentid          int,
strategycode          varchar(20),
`signal`              int,
FOREIGN KEY (`date`) REFERENCES dbo_datedim(`date`),
FOREIGN KEY (instrumentid) REFERENCES dbo_instrumentmaster(instrumentid),
FOREIGN KEY (strategycode) REFERENCES dbo_strategymaster(strategycode)
)
;

DROP TABLE IF EXISTS dbo_statisticalreturns;
CREATE TABLE dbo_statisticalreturns(           
`date`                 date,
instrumentid           int,
strategycode           varchar(20),
positionsize           int,
cashonhand             float,
portfoliovalue         float,
FOREIGN KEY (`date`) REFERENCES dbo_datedim(`date`),
FOREIGN KEY (instrumentid) REFERENCES dbo_instrumentmaster(instrumentid),
FOREIGN KEY (strategycode) REFERENCES dbo_strategymaster(strategycode)
)
;

DROP TABLE IF EXISTS dbo_engineeredfeatures;
CREATE TABLE dbo_engineeredfeatures(        
`date`                       date,   
instrumentid                 int,
rsi_14                       float,
macd_v                       float,
macds_v                      float,
boll_v                       float,
boll_ub_v                    float,
boll_lb_v                    float,
open_2_sma                   float,
wcma                         float,
scma                         float,
lcma                         float,
ltrough                      float,
lpeak                        float,
highfrllinelong              float,
medfrllinelong               float,
lowfrllinelong               float,
strough                      float,
speak                        float,
ktrough                      float,
kpeak                        float,
sema                         float,
mema                         float,
lema                         float,
volume_delta                 float,
FOREIGN KEY (`date`) REFERENCES dbo_datedim(`date`),
FOREIGN KEY (instrumentid) REFERENCES dbo_instrumentmaster(instrumentid)
)
;

DROP TABLE IF EXISTS dbo_macroeconalgorithmforecast;
CREATE TABLE dbo_macroeconalgorithmforecast(
forecastdate            date,
instrumentid            int,
macroeconcode			varchar(50),
forecastprice           float,
algorithmcode           varchar(50),
prederror               float,
FOREIGN KEY (forecastdate) REFERENCES dbo_datedim(`date`),
FOREIGN KEY (instrumentid) REFERENCES dbo_instrumentmaster(instrumentid),
FOREIGN KEY (algorithmcode) REFERENCES dbo_algorithmmaster(algorithmcode)
)
;

DROP TABLE IF EXISTS dbo_tempvisualize;
CREATE TABLE dbo_tempvisualize(
forecastdate			date,
instrumentid			int,
forecastcloseprice		float,
algorithmcode			varchar(50),
FOREIGN KEY (forecastdate) REFERENCES dbo_datedim(`date`),
FOREIGN KEY (instrumentid) REFERENCES dbo_instrumentmaster(instrumentid),
FOREIGN KEY (algorithmcode) REFERENCES dbo_algorithmmaster(algorithmcode)
)
;

DROP TABLE IF EXISTS dbo_paststatistics;
CREATE TABLE dbo_paststatistics(        
`date`                  date,   
high                    float,
low                     float,
`open`                  float,
`close`                 float,
volume                  float,
`adj close`             float,
instrumentid            int,
FOREIGN KEY (`date`) REFERENCES dbo_datedim(`date`),
FOREIGN KEY (instrumentid) REFERENCES dbo_instrumentmaster(instrumentid)
)
;
