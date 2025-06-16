from utils.loggerinfo import get_logger
from utils.earlystopping import EarlyStopping
from datamaker import generate_dataloader
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from model_contrast import RLPnet
import numpy as np
import torch
from torch import optim
from torch import nn



class Engine():
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger('LSTM.txt', 'LSTM')
        self.model = RLPnet(len(self.args.catelist), self.args.predict_seq_window,
        self.args.season_seq_window!=0, self.args.trend_seq_window!=0)
        self.model.to(self.device)
        self.broad_point=self.args.mountain_height
        self.waiting = self.args.mountain_patience

    def train_one_epoch(self, train_loader, optimizer, criterion):
        model = self.model.train()
        train_loss, itercnt = 0.0, 0
        for _,  (batch_x, batch_y) in enumerate(train_loader):
            x = [xi.to(self.device).float() for xi in batch_x]
            y = batch_y.to(self.device).float().squeeze(2)
            y1 = y[:,:,0]
            y2 = y[:,:,1]
            y3 = y[:,:,2]
            y4 = y[:,:,3]

            single_batch_size = y1.shape[0]

            y_pred1, y_pred2, y_pred3, y_pred4 = model(x)

            optimizer.zero_grad()

            loss1 = criterion(y_pred1, y1)
            loss2 = criterion(y_pred2, y2)
            loss3 = criterion(y_pred3, y3)
            loss4 = criterion(y_pred4, y4)

            miner=min(1,(loss4/(loss1+loss2+loss3)).item())
            w1=((loss2+loss3)/2/(loss1+loss2+loss3)).item()
            w2=((loss1+loss3)/2/(loss1+loss2+loss3)).item()
            w3=((loss1+loss2)/2/(loss1+loss2+loss3)).item()

            loss = miner*w1*loss1+miner*w2*loss2+miner*w3*loss3+loss4
            miner = min(1, (loss1 / (loss2 + loss3)).item())
            w1 = (loss3 / (loss2 + loss3)).item()
            w2 = (loss2 / (loss2 + loss3)).item()
            loss = loss1 + miner * w1 * loss2 + miner * w2 * loss3

            # loss = loss1 + loss2 + loss3 + loss4

            loss.backward()
            optimizer.step()
            train_loss += loss * single_batch_size
            itercnt += single_batch_size
            print(">>> Training [{}/{} ({:.2f}%)] MSELoss:{:.6f}".format(
                itercnt, len(train_loader.dataset), 100.0 * itercnt / len(train_loader.dataset), loss.item()), end="\r"
            )
        print("")
        train_loss /= len(train_loader.dataset)
        return train_loss.item()


    def validate_one_epoch(self, valid_loader, criterion):
        model = self.model.eval()
        valid_loss, itercnt = 0.0, 0
        with torch.no_grad():
            for _,  (batch_x, batch_y) in enumerate(valid_loader):
                x = [xi.to(self.device).float() for xi in batch_x]
                y = batch_y.to(self.device).float().squeeze(2)
                y1 = y[:, :, 0]
                y2 = y[:, :, 1]
                y3 = y[:, :, 2]
                y4 = y[:, :, 3]
                single_batch_size = y1.shape[0]
                y_pred1, y_pred2, y_pred3, y_pred4 = model(x)
                loss1 = criterion(y_pred1, y1)
                loss2 = criterion(y_pred2, y2)
                loss3 = criterion(y_pred3, y3)
                loss4 = criterion(y_pred4, y4)

                loss = loss1 + loss2 + loss3 + loss4

                valid_loss += loss * single_batch_size
                itercnt += single_batch_size
                print(">>> Validating [{}/{} ({:.2f}%)] MSELoss:{:.6f}".format(
                    itercnt, len(valid_loader.dataset), 100.0 * itercnt / len(valid_loader.dataset), loss.item()), end="\r"
                )
            print("")
            valid_loss /= len(valid_loader.dataset)
        return valid_loss.item()

    def test(self, test_loader):
        model = self.model.eval()
        model.load_state_dict(torch.load("checkpoint.pth"))
        prediction1, groundtruth1,prediction2, groundtruth2,prediction3, groundtruth3,prediction4, groundtruth4, itercnt = [],[],[],[],[],[],[],[], 0
        with torch.no_grad():
            for _, (batch_x, batch_y) in enumerate(test_loader):
                x = [xi.to(self.device).float() for xi in batch_x]
                y = batch_y.to(self.device).float().squeeze(2)
                single_batch_size = y.shape[0]

                y_true=y.cpu().data.numpy()
                y_pred1, y_pred2, y_pred3, y_pred4 = model(x)
                y_pred = torch.stack([y_pred1, y_pred2, y_pred3, y_pred4],2).cpu().data.numpy()
                y_true=y_true.squeeze(0)
                y_pred = y_pred.squeeze(0)

                inversed_y_true = test_loader.dataset.inverse_transform(y_true)#[0].reshape(-1, 1))
                inversed_y_pred = test_loader.dataset.inverse_transform(y_pred)#[0].reshape(-1, 1))

                inversed_y_true1 = inversed_y_true[:,0]
                inversed_y_true2 = inversed_y_true[:,1]
                inversed_y_true3 = inversed_y_true[:,2]
                inversed_y_true4 = inversed_y_true[:,3]
                inversed_y_pred1 = inversed_y_pred[:,0]
                inversed_y_pred2 = inversed_y_pred[:,1]
                inversed_y_pred3 = inversed_y_pred[:,2]
                inversed_y_pred4 = inversed_y_pred[:,3]

                prediction1.append(inversed_y_pred1)
                groundtruth1.append(inversed_y_true1)

                prediction2.append(inversed_y_pred2)
                groundtruth2.append(inversed_y_true2)

                prediction3.append(inversed_y_pred3)
                groundtruth3.append(inversed_y_true3)

                prediction4.append(inversed_y_pred4)
                groundtruth4.append(inversed_y_true4)


                itercnt += single_batch_size
                print(">>> Testing [{}/{} ({:.2f}%)]".format(
                    itercnt, len(test_loader.dataset),
                    100.0 * itercnt / len(test_loader.dataset)), end="\r"
                )

        rmse_error1 = np.sqrt(mean_squared_error(np.asarray(prediction1), np.asarray(groundtruth1)))
        mea_error1 = mean_absolute_error(np.asarray(prediction1), np.asarray(groundtruth1))
        # return rmse_error1.item(),mea_error1.item()
        # self.logger.info('ALL RMSE 冰箱: \t\t{:.6f}'.format(rmse_error1.item()))
        # self.logger.info('ALL MEA 冰箱: \t\t{:.6f}'.format(mea_error1.item()))

        rmse_error2 = np.sqrt(mean_squared_error(np.asarray(prediction2), np.asarray(groundtruth2)))
        mea_error2 = mean_absolute_error(np.asarray(prediction2), np.asarray(groundtruth2))
        # return rmse_error2.item(),mea_error2.item()

        # self.logger.info('ALL RMSE 空调: \t\t{:.6f}'.format(rmse_error2.item()))
        # self.logger.info('ALL MEA 空调: \t\t{:.6f}'.format(mea_error2.item()))
        #
        rmse_error3 = np.sqrt(mean_squared_error(np.asarray(prediction3), np.asarray(groundtruth3)))
        mea_error3 = mean_absolute_error(np.asarray(prediction3), np.asarray(groundtruth3))
        # return rmse_error3.item(), mea_error3.item()
        #
        # self.logger.info('ALL RMSE 洗衣机: \t\t{:.6f}'.format(rmse_error3.item()))
        # self.logger.info('ALL MEA 洗衣机: \t\t{:.6f}'.format(mea_error3.item()))
        #
        rmse_error4 = np.sqrt(mean_squared_error(np.asarray(prediction4), np.asarray(groundtruth4)))
        mea_error4 = mean_absolute_error(np.asarray(prediction4), np.asarray(groundtruth4))
        return rmse_error1.item(),mea_error1.item(),rmse_error2.item(),mea_error2.item(),rmse_error3.item(),mea_error3.item(),rmse_error4.item(), mea_error4.item()
        #
        # self.logger.info('ALL RMSE 电视: \t\t{:.6f}'.format(rmse_error4.item()))
        # self.logger.info('ALL MEA 电视: \t\t{:.6f}'.format(mea_error4.item()))

    def train(self):
        train_loader = generate_dataloader(self.args, 'train')
        valid_loader = generate_dataloader(self.args, 'valid')
        test_loader = generate_dataloader(self.args, 'test')
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        criterion = nn.MSELoss()
        optimizer_smaller = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate*self.args.mountain_decay, weight_decay=self.args.weight_decay)      #10       #1000
        epochs = self.args.train_epochs
        flag = 0
        count = 0
        count_sum=0
        mountain_top = 0
        # increse_flag = 0    #空调
        train_losses, valid_losses = [], []
        for epoch in range(epochs):
            if flag==0:
                train_loss = self.train_one_epoch(train_loader, optimizer, criterion)
                valid_loss = self.validate_one_epoch(valid_loader, criterion)
                mountain_top = valid_loss
            else:
                print("use small lr")
                count_sum+=1
                torch.save(self.model.state_dict(), "temp_check.pth")
                train_loss = self.train_one_epoch(train_loader, optimizer_smaller, criterion)
                valid_loss = self.validate_one_epoch(valid_loader, criterion)
            if flag==1 and mountain_top < valid_loss:
                count+=1
            if valid_loss<self.broad_point:
                flag=1
            if count>= self.waiting[0] or count_sum>self.waiting[1]:
                self.model.eval().load_state_dict(torch.load("temp_check.pth"))
                flag=0
                count=0



            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            # self.logger.info("Train Epoch[{}/{}]: Average Train MSELoss: {:.6f} | Average Validate MSELoss: {:.6f}".format(
            #     epoch, epochs, train_loss, valid_loss))


            # if epoch>2:     #空调
            #     if train_loss>train_losses[epoch-1] and valid_loss>valid_losses[epoch-1]:
            #         increse_flag+=1
            #     else:
            #         increse_flag=0
            early_stopping(valid_loss, self.model)
            # if epoch>100 and valid_loss<0.007:      #冰箱
            # # # if early_stopping.early_stop or valid_loss<0.003:
            #     self.logger.info("Early Stopped!")
            #     rmse, mea = self.test(test_loader)
            #     self.logger.info('ALL RMSE 冰箱: \t\t{:.6f}'.format(rmse))
            #     self.logger.info('ALL MEA 冰箱: \t\t{:.6f}'.format(mea))
            #     break

            # if increse_flag>=3:      #空调
            #     self.logger.info("Early Stopped!")
            #     rmse, mea = self.test(test_loader)
            #     self.logger.info('ALL RMSE 空调: \t\t{:.6f}'.format(rmse))
            #     self.logger.info('ALL MEA 空调: \t\t{:.6f}'.format(mea))
            #     break

            # if epoch>100 and valid_loss<0.007:      #洗衣机
            #     self.logger.info("Early Stopped!")
            #     rmse, mea = self.test(test_loader)
            #     self.logger.info('ALL RMSE 空调: \t\t{:.6f}'.format(rmse))
            #     self.logger.info('ALL MEA 空调: \t\t{:.6f}'.format(mea))
            #     break

            rmse1,mea1,rmse2,mea2,rmse3,mea3,rmse4,mea4=self.test(test_loader)
            # self.logger.info('ALL RMSE 冰箱: \t\t{:.6f}'.format(rmse))
            # self.logger.info('ALL MEA 冰箱: \t\t{:.6f}'.format(mea))
            # if valid_loss<0.04:
            #     self.logger.info('ALL RMSE 空调: \t\t{:.6f}'.format(rmse))
            #     self.logger.info('ALL MEA 空调: \t\t{:.6f}'.format(mea))
            #     self.logger.info(epoch)
            #     break

            self.logger.info('ALL RMSE 冰箱: \t\t{:.6f}'.format(rmse1))
            self.logger.info('ALL MEA 冰箱: \t\t{:.6f}'.format(mea1))
            self.logger.info('ALL RMSE 空调: \t\t{:.6f}'.format(rmse2))
            self.logger.info('ALL MEA 空调: \t\t{:.6f}'.format(mea2))
            self.logger.info('ALL RMSE 洗衣机: \t\t{:.6f}'.format(rmse3))
            self.logger.info('ALL MEA 洗衣机: \t\t{:.6f}'.format(mea3))
            self.logger.info('ALL RMSE 电视: \t\t{:.6f}'.format(rmse4))
            self.logger.info('ALL MEA 电视: \t\t{:.6f}'.format(mea4))

        self.test(test_loader)