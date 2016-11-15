import datetime
import time
import dateutil.relativedelta
import logging

class Duration:
    def start(self):
        logging.debug('start timer')
        self.startTime = datetime.datetime.now()
        return
    def end(self, num_epoch = 1, num_iteration=1):
        self.endTime = datetime.datetime.now()
        logging.debug('end timer')
        self.dispDuration(num_epoch, num_iteration)
        return
    def dispDuration(self, num_epoch, num_iteration):
        rd = dateutil.relativedelta.relativedelta (self.endTime , self.startTime)
        logging.debug( "Duration: %d:%d:%d One epoch %f minutes, One iteration %f minutes" % (rd.hours, rd.minutes, rd.seconds, 
                                                                                              (rd.hours *60.0 + rd.minutes + rd.seconds/60.0)/num_epoch,
                                                                                              (rd.hours *60.0 + rd.minutes+ rd.seconds/60.0)/num_iteration))
#         logging.debug "Duration: %d years, %d months, %d days, %d hours, %d minutes and %d seconds" \
#         % (rd.years, rd.months, rd.days, rd.hours, rd.minutes, rd.seconds)
            
        
        return
    
if __name__ == "__main__":   
    obj= Duration()
    obj.start()
    time.sleep(3)
    obj.end()