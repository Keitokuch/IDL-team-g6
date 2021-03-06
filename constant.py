LETTER_LIST = ['<sos>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
         'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ', '<eos>']

MIN_LABEL_CNT=5

label_lists = {}
label_lists[0] = ['Saya', 'Customers+Taki', 'BystanderE', 'TrainAnnouncer', 'Shinta', 'Mitsuha+Teshi+Saya', 'StudentA', 'BystanderB', 'StudentD', 'MitsuhaMom', 'Yotsuha+Mitsuha+Kids', 'Kitchen',
              'Yotsuha+Mitsuha', 'Yotsuha', 'Foreman+TeshiMom', 'Yotsuha+Grandma', 'Waitress+Chef', 'Okudera+Staff', 'Saya+Mistuha', 'Mitsuha+Teshi', 'Okudera', 'Customers', 'Radio', 'Tsukasa+Taki',
              'Okudera+Tsukasa', ' Mitsuha', 'Futaba', 'BystanderF', 'Foreman', 'Sara+Mitsuha', 'Taki+Tsukasa', 'Takagi', 'TeshiDad', 'Mitsuha+TV', 'Toshiki+Mitsuha', 'Teacher', 'StudentD+Class', 'Mitsuha+Taki',
              'Shinta+Taki', 'Chef', 'Band', 'Customers+Okudera', 'Taki+Shinta', 'Grandma+Taki', 'Yatsuha+Mitsuha',
              'Grandma+Mitsuha', 'Kids', 'Conductor', 'Grandma', 'Teshi+Saya', 'TownHall', 'Teshi+TeshiMom', 'None',
              'Taki+Mitsuha', 'Saya+Teshi', 'BackCharA+BackCharB', 'Sara+Teshi', 'Staff', 'Mitsuha+Saya+Teshi', 'Crowd',
              'TV+Mitsuha', 'BystanderD', 'Teacher+Mitsuha', 'Taki+Staff', 'StudentB+StudentC', 'TV', 'StudentB', 'Shopkeeper',
              'TownHall+Crowd', 'Firefighters', 'Foreman+Teshi', 'Mitsuha+Kids', 'Sara', 'Phone', 'Tsukasa+Okudera', 'Teshi', 'Gradnma', 'Mitsuha+Grandma', 'StudentA+Mitsuha', 'Mitsuha+Yotsuha', 'Taki+Okudera', 'Tsukasa', 'Taki', 'TakiDad', 'Mitsuha', 'Shinta+Tsukasa',
              'Sara+Crowd', 'Toshiki+Foreman', 'Okudera+Taki', 'BystanderC', 'Teshi+Mitsuha', 'Toshiki+BystanderA',
              'Saya+Mitsuha', 'Waitress', 'TakiDad+Taki', 'Toshiki', 'Mitsuha+Futaba', 'Toshiki+Grandma', 'Class']

label_lists[5] = ['Tsukasa', 'None', 'Waitress', 'Teacher', 'Mitsuha+Yotsuha', 'Taki+Mitsuha', 'Okudera', 'Customers', 'Mitsuha+Teshi', 'Band', 'Crowd', 'Okudera+Taki', 'Sara', 'Staff', 'Shinta+Tsukasa', 'Mitsuha+Taki', 'Taki+Okudera', 'Saya+Teshi', 'Toshiki', 'Yotsuha', 'Radio', 'Taki+Tsukasa', 'Teshi', 'Mitsuha', 'Saya', 'Grandma', 'Teshi+Mitsuha', 'TV', 'Taki']

LABEL_LIST = label_lists[MIN_LABEL_CNT]

def set_min_label_count(min_cnt):
    global LABEL_LIST
    LABEL_LIST = label_lists[min_cnt]
