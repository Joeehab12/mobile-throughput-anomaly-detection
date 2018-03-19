install.packages("forecast")
install.packages("fpp2")
install.packages("tidyr")
install.packages("dplyr")
install.packages("lubridate")
install.packages("chron")
install.packages("forecast")
install.packages("fpp2")
install.packages("zoo")
install.packages("xts")
install.packages("aTSA")
install.packages(("lubridate"))
install.packages("dplyr")
install.packages("plyr")
install.packages("magrittr")
install.packages("data.table")

library(data.table)
library(lubridate)
library(magrittr)
library(plyr)

library(dplyr)
library(imputeTS)
library(forecast)
library(fpp2)
library(tidyr)
library(dplyr)
library(lubridate)
library(chron)
library(forecast)
library(fpp2)
library(zoo)
library(xts)
library(aTSA)


rm(list=ls())
#display work items
ls()
setwd("E:/mobile-throughput-anomaly-detection")
getwd()

data <- read.csv("data.csv")

# select columns for day, throughput value and Date Time 
interpolation_window <- subset(data, select=c("Wday","value", "DateTime","Day","Year","Month"))
# save date time values as POSIX variable
times.init <-as.POSIXct(strptime(interpolation_window[,3], '%m/%d/%Y  %H:%M'))
# put date time object aside Throuhput value in zoo object
data2 <-zoo(interpolation_window[,2],times.init)

# merge the previous series with missing values
data3 <-merge(data2, zoo(,seq(min(times.init),as.POSIXct("2018-01-09 23:45:00") , by = "15 mins")))
# converts zoo object to dataframe
data3<-fortify.zoo(data3)
# array to map dates to corresponding days
dates_map<-c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday","Friday", "Saturday")
# save converted dates in data frame
data7<-data.frame(dates_map[as.POSIXlt(data3$Index)$wday + 1])
# rename column names
colnames(data7) <- "Wday"
# merge date time and throuput value columns with day column
data8<- merge(zoo(data3),zoo(data7))
# converts zoo object to dataframe
data8<-fortify.zoo(data8)

colnames(data8)<- c("index","datetime","value","Wday")
#select certain days from the whole dataset
sundays_sub<- subset(data8,Wday == "Sunday")
mondays_sub<- subset(data8,Wday == "Monday")
tuesdays_sub<- subset(data8,Wday == "Tuesday")
wednesdays_sub<- subset(data8,Wday == "Wednesday")
thursdays_sub<- subset(data8,Wday == "Thursday")
fridays_sub<- subset(data8,Wday == "Friday")
saturdays_sub<- subset(data8,Wday == "Saturday")

#convert df to matrix where rows are number of points and columns are points
# for each day
mat_sunday <- matrix(sundays_sub$value,nrow = 96)
mat_monday <- matrix(mondays_sub$value,nrow = 96)
mat_tuesday <- matrix(tuesdays_sub$value,nrow = 96)
mat_wednesday <- matrix(wednesdays_sub$value,nrow = 96)
mat_thursday <- matrix(thursdays_sub$value,nrow = 96)
mat_friday <- matrix(fridays_sub$value,nrow = 96)
mat_saturday <- matrix(saturdays_sub$value,nrow = 96)

#initialize empty lists for each day
allSundayPoints <- list()
allMondayPoints <-list()
allTuesdayPoints<- list()
allWednesdayPoints<-list()
allThursdayPoints<- list()
allFridayPoints<- list()
allSaturdayPoints<-list()

#fill na values using linear interpolation 
for (i in 1:96)
{
  test1<-na.approx(mat_sunday[i,],rule=2)
  test2<-na.approx(mat_monday[i,],rule=2)
  test3<-na.approx(mat_tuesday[i,],rule=2)
  test4<-na.approx(mat_wednesday[i,],rule=2)
  test5<-na.approx(mat_thursday[i,],rule=2)
  test6<-na.approx(mat_friday[i,],rule=2)
  test7<-na.approx(mat_saturday[i,],rule=2)
  allSundayPoints[[i]] <- test1
  allMondayPoints[[i]] <- test2
  allTuesdayPoints[[i]] <-test3
  allWednesdayPoints[[i]]<-test4
  allThursdayPoints[[i]]<- test5
  allFridayPoints[[i]] <- test6
  allSaturdayPoints[[i]]<-test7
}
#unlist matrix then transpose to get points for each day
sun <- unlist(allSundayPoints)
mon <- unlist(allMondayPoints)
tues<-unlist(allTuesdayPoints)
wed<- unlist(allWednesdayPoints)
thur<- unlist(allThursdayPoints)
fri<- unlist(allFridayPoints)
sat<- unlist(allSaturdayPoints)

sun_interpolated<-matrix (sun, ncol = 96)
mon_interpolated<-matrix (mon, ncol = 96)
tues_interpolated<-matrix (tues, ncol = 96)
wed_interpolated<-matrix (wed, ncol = 96)
thur_interpolated<-matrix (thur, ncol = 96)
fri_interpolated<-matrix (fri, ncol = 96)
sat_interpolated<-matrix (sat, ncol = 96)

for (i in 1:74)
{
  write.table (sun_interpolated[i,],"test.csv",append = TRUE,col.names = FALSE,row.names = FALSE)
  write.table (mon_interpolated[i,],"test.csv",append = TRUE,col.names = FALSE,row.names = FALSE)
  write.table (tues_interpolated[i,],"test.csv",append = TRUE,col.names = FALSE,row.names = FALSE)
  if (i <= 73){
  write.table (wed_interpolated[i,],"test.csv",append = TRUE,col.names = FALSE,row.names = FALSE)
  write.table (thur_interpolated[i,],"test.csv",append = TRUE,col.names = FALSE,row.names = FALSE)
  write.table (fri_interpolated[i,],"test.csv",append = TRUE,col.names = FALSE,row.names = FALSE)
  write.table (sat_interpolated[i,],"test.csv",append = TRUE,col.names = FALSE,row.names = FALSE)
  }
}

interpolated_values<-read.csv("test.csv",header = FALSE)
date_time<-data8$datetime
wday<- data8$Wday
interpolated_values[2]<-date_time
interpolated_values[3]<-wday
colnames(interpolated_values)<- c("value","datetime","wday")
write.csv(interpolated_values,"interpolated.csv")
rownames(interpolated_values)<-interpolated_values$datetime

interpolated_values$value[]
interpolated_tuesdays<- subset(interpolated_values, as.character(datetime) >="2017-01-10 00:00:00" & as.character(datetime) < "2017-01-11 00:00:00" )
autoplot(ts(interpolated_tuesdays$value),ylab = "Mobile Throughput")
axis(interpolated_tuesdays$datetime)
