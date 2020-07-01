
/*SCRIPT TO GENERATE ALL THE BACK-END DATA TABLES FOR GM FINTECH APPLICATION */
/*TABLES ARE NOT DEPENDENT UPON EACH OTHER FOR THE BACK-END STRUCTURE        */
/*MISSING A TABLE COULD CAUSE ISSUES IN RUNNING THE PYTHON SCRIPTS AND       */
/*RUNNING TABLEAU MEASURES                                                   */
/*SCRIPT IS WRITTEN FOR SQL FORMAT ACCEPTED BY MYSQL 8.X                     */


use GMFSP_db;

DROP TABLE IF EXISTS dbo_strategymaster;
create table dbo_strategymaster 
(          
strategycode        varchar(20),
strategyname        varchar(50),
primary key (strategycode) 
)
;


DROP TABLE IF EXISTS dbo_algorithmforecast;
create table dbo_algorithmforecast 
(          
forecastdate            date,
instrumentid            int,
forecastcloseprice      float,
algorithmcode           varchar(50),
prederror               float,
primary key (forecastdate, instrumentid, algorithmcode)
)
;


DROP TABLE IF EXISTS dbo_actionsignals;
create table dbo_actionsignals 
(          
date                  date,
instrumentid          int,
strategycode          varchar(50),
`signal`              int,
primary key (date, instrumentid, strategycode)
)
;


DROP TABLE IF EXISTS dbo_algorithmmaster;
create table dbo_algorithmmaster 
(          
algorithmcode         varchar(50),
algorithmname         varchar(50),
primary key (algorithmcode)
)
;


drop table if exists dbo_instrumentmaster;  
create table dbo_instrumentmaster(
instrumentid            int,
instrumentname          varchar(50),
`type`                  varchar(50),
exchangename            varchar(50),
primary key (instrumentid)
)
;


DROP TABLE IF EXISTS dbo_instrumentstatistics;
create table dbo_instrumentstatistics(        
date                    date,   
high                    float,
low                     float,
`open`                  float,
`close`                 float,
volume                  float,
`adj close`             float,
instrumentid            int,
primary key (date, instrumentid)
)
;


DROP TABLE IF EXISTS dbo_datedim;
create table dbo_datedim(
`date`                 date,
`year`                 int,
`month`                int,
qtr                    int,
weekend                int,
isholiday              int,
primary key (date)
)
;


DROP TABLE IF EXISTS dbo_statisticalreturns;
create table dbo_statisticalreturns
(           
`date`                 date,
instrumentid           int,
strategycode           varchar(50),
positionsize           int,
cashonhand             float,
portfoliovalue         float,
primary key (date, instrumentid, strategycode)
)
;


DROP TABLE IF EXISTS dbo_engineeredfeatures;
create table dbo_engineeredfeatures(        
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
primary key (date, instrumentid)
)
;



/* MUST INSERT VALUES INTO INSTRUMENTMASTER TABLE IN ORDER FOR DATABASE TO WORK */
/* USE INSERT_INTO_INSTRUMENT_MASTER_MYSQL.sql FILE TO ADD SYMBOLS              */
/* USE REMOVE_FROM_INSTRUMENT_MASTER_MYSQL.sql FILE TO REMOVE SYMBOLS           */

TRUNCATE TABLE dbo_instrumentmaster;
insert into dbo_instrumentmaster
values (1 , 'GM'   , 'Equity' , 'YAHOO'),
	   (2 , 'PFE'  , 'Equity' , 'YAHOO'),
	   (3 , 'SPY'  , 'Equity' , 'YAHOO'),
	   (4 , 'XPH'  , 'Equity' , 'YAHOO'),
	   (5 , 'CARZ' , 'Equity' , 'YAHOO'),
       (6 , '^TYX' , 'Equity' , 'YAHOO'),
	(7, 'FCAU' , 'Equity' , 'YAHOO'),
	(8, 'TM' , 'Equity', 'YAHOO'),
	(9, 'F', 'Equity' , 'YAHOO'),
	(10, 'HMC' , 'Equity' , 'YAHOO')
;

drop table if exists dbo_macroeconmaster;  
create table dbo_macroeconmaster(
macroeconcode          varchar(10),
macroeconname          varchar(50),
accesssourcekey        varchar(50),
accesssource		varchar(50),
datecreated		 date,
activecode		varchar(10),
primary key (macroeconname)
);

insert into dbo_macroeconmaster
values ('GDP' , 'GDP'   , 'FRED/NGDPPOT', 'Quandl', 0, 'A'),
	   ('UR' , 'Unemployment Rate'  , 'USMISERY/INDEX', 'Quandl', 0, 'A'),
	   ('IR' , 'Inflation Rate'  , 'USMISERY/INDEX', 'Quandl', 0, 'A'),
	   ('MI' , 'Misery Index'  , 'USMISERY/INDEX', 'Quandl', 0, 'A'),
       ('TYX', '30 Year Bond Yield', '^TYX', 'Yahoo', 0, 'I'),
       ('COVI', 'Crude Oil ETF Volatility Index', 'OVXCLS', 'FRED', 0, 'A'),
       ('CPIUC', 'Consumer Price Index Urban Consumers', 'CPIAUCSL', 'FRED', 0, 'A'),
       ('FSI', 'Financial Stress Index', 'STLFSI', 'FRED', 0, 'A')
;

DROP TABLE IF EXISTS dbo_macroeconstatistics;
CREATE TABLE dbo_macroeconstatistics (
	date	date,
	statistics	int,
    macroeconcode	varchar(10),
	primary key (macroeconcode)
    );

DROP TABLE IF EXISTS dbo_macroeconalgorithmforecast;
CREATE TABLE dbo_macroeconalgorithmforecast(
forecastdate            date,
instrumentid            int,
macroeconcode		varchar(10),
forecastprice           float,
algorithmcode           varchar(50),
prederror               float,
primary key (instrumentid)
);

DROP TABLE IF EXISTS dbo_tempvisualize;
CREATE TABLE dbo_tempvisualize(
	forecastdate	date,
    instrumentid	int,
    forecastcloseprice	float,
    algorithmcode	varchar(50)
);

DROP TABLE IF EXISTS dbo_paststatistics;
create table dbo_paststatistics(        
date                    date,   
high                    float,
low                     float,
`open`                  float,
`close`                 float,
volume                  float,
`adj close`             float,
instrumentid            int
)