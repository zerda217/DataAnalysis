### 1. 파일 불러오기 ####
# https://www.kaggle.com/dignil/predicting-heart-disease-decision-trees-and-nns
df <- read.csv('heart.csv') #파일 넣기
summary(df) # 내용 요약 보기
colnames(df)[1] <- 'age' # 변수 이름 바꾸기
colnames(df)

### 변수명 정의 ####
# age : 나이 | sex : 성별 (0:여자 1:남자)
# cp : 흉통 chest pain(1: 전형적 협심증 2: 비전형 협심증 3: 협심증이 아닌 통증 4: 무증상)
# trestbps : 안정시 혈압 분포
# chol : 콜레스테롤 | fbs : 공복혈당수치 (120이하 0:F 1:T)
# restecg : 안정시 심전도 측정(0:보통 1:ST-T파 이상 2:좌심실 비대)
# thalach : 최대 심박수
# exang : 통증을 포함한 운동(0:없음 1:있음) 
# oldpeak : 휴식과 관련된 운동으로 유발된 ST 우울증
# slope : 운동의 기울기(1: 오르막 2: 평평 3: 내리막) 
# ca : 주요 혈관의 수 (0: 1: 2: 3:)
# thal : | target : 심장병 유무 (0:없음 1:있음)

### 2. 변수 factor 로 변환 ####
str(df) # 변수 유형 확인
for(i in c(2,3,6,9,14))
{df[,i] <- as.factor(df[,i])}
summary(df)

### 3. 데이터 시각화 ####
x <- colnames(df) #변수 x에 데이터 넣기

#히스토그램
library(ggplot2)
options(repr.plot.width = 15, repr.plot.height = 12)
ggplot(df,aes(x = age , fill = target))+geom_histogram() #연령 분포
ggplot(df,aes(x = thalach , fill = target))+geom_histogram() #최대 심박수 분포
ggplot(df,aes(x = trestbps , fill = target))+geom_histogram() #안정시 혈압 분포

#범주형변수 상자그림
options(repr.plot.width = 15, repr.plot.height = 12)
ggplot(df,aes(x = age ,y= target , fill = fbs))+geom_boxplot() #상자 수염그림
ggplot(df,aes(y = thalach , x = cp ))+geom_jitter()+geom_violin(alpha=0.5) #바이올린 그림

#산점도 및 거품형 차트
options(repr.plot.width = 15, repr.plot.height = 12)
ggplot(df,aes(x = age ,y= thalach , size = chol))+scale_size(range = c(1,10))+geom_point(alpha = 0.5)+geom_smooth() #bubble plot to describe the relationship between age, maximum hear rate and cholesterol
ggplot(df,aes(x = age ,y= thalach , size = chol, color = fbs))+scale_size(range = c(1,20))+geom_point(alpha = 0.5)+geom_smooth()

#pairsplot
options(repr.plot.width = 30, repr.plot.height = 50)
#install.packages('GGally')
library(GGally)
ggpairs(df, aes(colour = target, alpha = 0.5))

library(corrplot)
corrplot(cor(df[,-c(2,3,6,9,14)]))

### 4. 데이터 결정모형 ####
#Train-Test split
#install.packages('caTools')
library(caTools)
set.seed(123) #랜덤값을 가질 수 있도록 기준점 설정, 없으면 할 때마다 다른 결과
index <- sample.split(df$target,SplitRatio = 0.8)
train <- subset(df,index ==TRUE) #학습용
test <- subset(df,index ==FALSE) #검증용

#Decision Tree Model
#install.packages('rpart.plot')
library(rpart)#for rpart decision tree
library(rpart.plot)#for visualising the tree
model.tree <- rpart(target~.,data = df)
rpart.plot(model.tree)
summary(model.tree)
confusionMatrix(as.factor
                (predict(model.tree,newdata = test, type = 'class')),test$target)

# 오분류율
yhat <- predict(model.tree, newdata=test, type="class")
ytest <- test$target
table(yhat,ytest)
cat("오분류율=", mean(yhat!=ytest)*100,"%")

library(tree)
model.tree2 <- tree(target~.,data = df)
plot(model.tree2)
text(model.tree2)

### 4-2. 신경망 모델 ####
#Single Layer Neural Network Model
#install.packages('NeuralNetTools')
library(nnet)#for neural netwrok model
library(NeuralNetTools)#for neural network plot
model.nn<-nnet(target~.,data = df, size = 5 , maxit = 1000)
plotnet(model.nn)

### 5. 모델 비교 ####
library(caret)
print('Decision Tree Model Accuracy')
confusionMatrix(as.factor
                (predict(model.tree,newdata = test, type = 'class')),test$target)
print('Neural Network Accuracy')
confusionMatrix(as.factor(predict(model.nn,newdata = test, type = 'class')),test$target)


### 6. 추가한 것, 트리모델####
#install.packages("C50")
library(C50)
tree_train <- C5.0(train[-14], train$target)
summary(tree_train)
tree_pred <- as.factor(predict(tree_train,test)) #모델평가
tree_pred
#install.packages("gmodels") #크로스테이블 생성
library(gmodels)
CrossTable(test$target,tree_pred,
           prop.chisq=FALSE, prop.c=FALSE, prop.r=FALSE,
           dnn=c("actual default", "predict default")) #모델을 통한 예측값
confusionMatrix(tree_pred, test$target)


#가지치기
#install.packages("tree")
library(tree)
treemod <- tree(target~. , data=train)
plot(treemod)
text(treemod)
tree_cut <- cv.tree(treemod, FUN=prune.misclass)
plot(tree_cut)
prune.trees <- prune.misclass(treemod, best=8)
plot(prune.trees)
text(prune.trees, pretty=0)

# 모델링 정확성 평가
library(e1071)
library(caret)
tree_pred2 <- as.factor(predict(prune.trees, test, type='class'))
confusionMatrix(tree_pred2, test$target)

summary(prune.trees)
