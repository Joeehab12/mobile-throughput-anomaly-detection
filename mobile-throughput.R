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

#why histogram doesnot show 17.840 ?
summary(data$value)
hist(data$value)
hist(data$value, breaks=12, col="red") 
boxplot(data$value)

d <- density(data$value) # returns the density data
plot(d) # plots the results 
polygon(d, col="red", border="blue") 

#
month<-subset(data,data$Year=="2016")
ggplot(data,aes(y = value, x = Date)) + geom_point(aes(color = Month))





#date column
my_data4 <- unite(data,col = "Date",Day, Month,Year,sep = "/")
write.table(my_data4$Date, file = "dateonly.csv",row.names=FALSE, na="",col.names=FALSE)

#date&time column
data8 <- with(data, as.POSIXct(paste(Date, Hour), format="%d/%m/%Y %H:%M"))
write.table(data8, file = "data25.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")


rdate<-as.Date(data$Date,"%d/%m/%y")
plot(data$value~rdate,type="l",col="red",axes="F")
box()
axis(1,rdate,format(rdate,"%m/%y"))

timeDate <- as.POSIXct(data$DateTime, format="%d/%m/%Y  %H:%M")   

temp<-subset(data ,data$Year=="2016" &data$Month=="8")
p<-ts(data$value,start=c(2016,8,14),end=c(2016,8,15),frequency=24*4)
#fix(p)
autoplot(p)





dts <- data$Date[1]
dts
# [1] 02/27/92 02/27/92 01/14/92 02/28/92 02/01/92
tms <- data$Hour[1]
tms
# [1] 23:03:20 22:29:56 01:03:30 18:21:03 16:56:26
x <- chron(dates = dts, times = tms, format = c(dates = "d/m/y", times = "h:m:s"))
x
x <- chron(dates = dts, times = tms, format = c(dates = "d/m/y", times = "h:m:s"))

plot(data$value~x,type="l",col="red",axes="F")

Lines<-read.csv("data2.csv")




test <- subset(data ,data$Year=="2017")

timeDate <- as.POSIXct(test$DateTime, format="%m/%d/%Y  %H:%M")   

mydata <- data.frame(value=test$value,timestamps=timeDate)
mydata1 <- xts(mydata$value, order.by=mydata$timestamps)

rdate<-as.Date(data$Date,"%d/%m/%y")
ax <- as.POSIXct(test$DateTime, format="%m/%d")
plot(as.ts(mydata1)) # Decompose after conversion to ts
#axis(1,at=ax,labels=format(ax,"%m/%d/%Y"))
axis(1,rdate,format(rdate,"%y"))

data1<-subset(data ,data$Year=="2016" &(data$Month=="12" |data$Month=="11"))#&data$Day=="1")
data1<-subset(data ,data$Year=="2016" &data$Month=="12" & data$Day=="1")#&data$Day=="1")

timeDate <- as.POSIXct(data1$DateTime, format="%m/%d/%Y  %H:%M")   
mydata <- data.frame(value=data1$value, timestamps=timeDate)
mydata <- xts(mydata$value, order.by=mydata$timestamps)

attr(mydata, 'frequency') <- 24*4  # Set the frequency of the xts object to weekly
periodicity(mydata)             # check periodicity: weekly 
plot(decompose(as.ts(mydata)))  # Decompose after conversion to ts
try<-stl(mydata, "periodic")
plot(try)
adf.test(data$value)





Day1<-subset(data ,data$Year=="2016" &data$Month=="12")#&data$Day=="1")

timeDate <- as.POSIXct(Day1$DateTime, format="%m/%d/%Y  %H:%M")

ax <- as.POSIXct(Day1$DateTime, format="%m/%d/%Y")


#Days
Day1$value
plot(Day1$value~timeDate,type="l",col="red",axes="F")
axis(1,at=ax,labels=format(ax,"%d"))

par(new=TRUE)
plot(Day2$value~timeDate1,type="l",col="black",axes="F")
par(new=TRUE)
plot(Day3$value~timeDate2,type="l",col="blue",axes="F")








Year<-subset(data ,data$Year=="2016" )
Year1<-subset(data ,data$Year=="2017" )
Year2<-subset(data ,data$Year=="2018" )

timeDateY <- as.POSIXct(Year$DateTime, format="%m/%d/%Y  %H:%M")
timeDateY1 <- as.POSIXct(Year1$DateTime, format="%m/%d/%Y  %H:%M")
timeDateY2 <- as.POSIXct(Year2$DateTime, format="%m/%d/%Y  %H:%M")


plot(Year1$value~timeDateY1,type="l",col="red",axes="F")
axis(1,at=timeDateY1,labels=format(timeDateY1,"%m"))

par(new=TRUE)
plot(Year1$value~timeDateY1,type="l",col="black",axes="F",add=TRUE)
par(new=TRUE)
plot(Year2$value~timeDateY2,type="l",col="blue",axes="F",add=TRUE)

acf(data$value)
pacf(data$value)

# select columns for day, throughput value and Date Time 
interpolation_window <- subset(data, select=c("Wday","value", "DateTime"))
# save date time values as POSIX variable
times.init <-as.POSIXct(strptime(interpolation_window[,3], '%m/%d/%Y  %H:%M'))
# put date time object aside Throuhput value in zoo object
data2 <-zoo(interpolation_window[,2],times.init)
# merge the previous series with missing values
data3 <-merge(data2, zoo(,seq(min(times.init), max(times.init), by = "15 mins")))
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

#select certain days from the whole dataset
sundays_sub<- subset(data8,Wday == "Sunday")
mondays_sub<- subset(data8,Wday == "Monday")
tuesdays_sub<- subset(data8,Wday == "Tuesday")
wednesdays_sub<- subset(data8,Wday == "Wednesday")
thursdays_sub<- subset(data8,Wday == "Thursday")
fridays_sub<- subset(data8,Wday == "Friday")
saturdays_sub<- subset(data8,Wday == "Saturday")

# convert values from character to numeric "double"
sundays<-as.numeric(as.character(sundays_sub[,3]))
# interpolate NA values
sundays_interpolated<-na.interpolation(sundays,option = "linear")
# replace old column with interpolation data
sundays_sub[,3]<-sundays_interpolated

mondays<-as.numeric(as.character(mondays_sub[,3]))
mondays_interpolated<-na.interpolation(mondays,option = "linear")
mondays_sub[,3]<-mondays_interpolated

tuesdays<-as.numeric(as.character(tuesdays_sub[,3]))
tuesdays_interpolated<-na.interpolation(tuesdays,option = "linear")
tuesdays_sub[,3]<-tuesdays_interpolated

wednesdays<-as.numeric(as.character(wednesdays_sub[,3]))
wednesdays_interpolated<-na.interpolation(wednesdays,option = "linear")
wednesdays_sub[,3]<-wednesdays_interpolated

thursdays<-as.numeric(as.character(thursdays_sub[,3]))
thursdays_interpolated<-na.interpolation(thursdays,option = "linear")
thursdays_sub[,3]<-thursdays_interpolated

fridays<-as.numeric(as.character(fridays_sub[,3]))
fridays_interpolated<-na.interpolation(fridays,option = "linear")
fridays_sub[,3]<-fridays_interpolated

saturdays<-as.numeric(as.character(saturdays_sub[,3]))
saturdays_interpolated<-na.interpolation(saturdays,option = "linear")
saturdays_sub[,3]<-saturdays_interpolated

# concatenate each single day dataframes to form one dataframe 
test<-rbind(mondays_sub,sundays_sub)
test2<-rbind(test,tuesdays_sub)
test3<-rbind(test2,wednesdays_sub)
test4<-rbind(test3,thursdays_sub)
test5<-rbind(test4,fridays_sub)
test6<-rbind(test5,saturdays_sub)

# sort dataframe by index to maintain original order
st<-test6[order(test6$Index),]
# rename column names
setnames(st,old = c("Index","Index.1","data3","Wday"),new = c("Id","Date Time","Value","Day"))

write.csv(st,"interpolated.csv")
