import numpy as np
import os
import struct
import cv2
# import easygui
import csv

def getDVSeventsDavis(file, ROI=np.array([]), numEvents=1e10, startEvent=0):
    print('\ngetDVSeventsDavis function called \n')
    sizeX = 346
    sizeY = 260
    x0 = 0
    y0 = 0
    x1 = sizeX
    y1 = sizeY
    if len(ROI) != 0:
        if len(ROI) == 4:
            print('Region of interest specified')
            x0 = ROI(0)
            y0 = ROI(1)
            x1 = ROI(2)
            y1 = ROI(3)
        else:
            print(
                'Unknown ROI argument. Call function as: \n getDVSeventsDavis(file, ROI=[x0, y0, x1, y1], numEvents=nE, startEvent=sE) '
                'to specify ROI or\n getDVSeventsDavis(file, numEvents=nE, startEvent=sE) to not specify ROI')
            return

    else:
        print('No region of interest specified, reading in entire spatial area of sensor')

    print('Reading in at most', str(numEvents))
    print('Starting reading from event', str(startEvent))

    triggerevent = int('400', 16)
    polmask = int('800', 16)
    xmask = int('003FF000', 16)
    ymask = int('7FC00000', 16)
    typemask = int('80000000', 16)
    typedvs = int('00', 16)
    xshift = 12
    yshift = 22
    polshift = 11
    x = []
    y = []
    ts = []  # Timestamps tick is 1 us
    pol = []
    numeventsread = 0

    length = 0
    aerdatafh = open(file, 'rb')
    k = 0
    p = 0
    statinfo = os.stat(file)
    if length == 0:
        length = statinfo.st_size
    print("file size", length)

    lt = aerdatafh.readline()
    while lt and str(lt)[2] == "#":
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        continue

    aerdatafh.seek(p)
    tmp = aerdatafh.read(8)
    p += 8
    while p < length:
        ad, tm = struct.unpack_from('>II', tmp)
        ad = abs(ad)
        if (ad & typemask) == typedvs:
            xo = sizeX - 1 - float((ad & xmask) >> xshift)
            yo = float((ad & ymask) >> yshift)
            polo = 1 - float((ad & polmask) >> polshift)
            if xo >= x0 and xo < x1 and yo >= y0 and yo < y1:
                x.append(xo)
                y.append(yo)
                pol.append(polo)
                ts.append(tm)
        aerdatafh.seek(p)
        tmp = aerdatafh.read(8)
        p += 8
        numeventsread += 1

    ts[:] = [x - ts[0] for x in ts]  # absolute time -> relative time
    x[:] = [int(a) for a in x]
    y[:] = [int(a) for a in y]

    print('Total number of events read =', numeventsread)
    print('Total number of DVS events returned =', len(ts))

    return ts, x, y, pol


def event_neighbor_filter(data=np.array([]), height=260, width=346, margin=1, threshold=2):
    img = np.zeros([height, width], dtype=np.int8)

    for idx in range(0, data.shape[0]):
        img[data[idx, 1], data[idx, 0]] = 1

    pos_tuple = np.where(img == 1)
    pos = np.array([pos_tuple[0], pos_tuple[1]]).T

    img_padding = np.zeros([height + 2 * margin, width + 2 * margin], dtype=np.int8)
    img_padding[margin:height + margin, margin:width + margin] = img

    for idx in range(0, pos.shape[0]):
        num_of_events = 0
        for i in range(-margin, margin + 1):
            for j in range(-margin, margin + 1):
                num_of_events += img_padding[pos[idx][0] + i][pos[idx][1] + j]
        img[pos[idx][0]][pos[idx][1]] = num_of_events > threshold

    data_filtered_tuple = np.where(img == 1)
    data_filtered = np.array([data_filtered_tuple[1], data_filtered_tuple[0]]).T

    return data_filtered


if __name__ == "__main__":
    binary = False
    filter_flag = False   # True : use filter
    sub_num = 42         # subject number of filtration 
    st_sub_idx = 1       # start subject index of filtration
    sub_group = 8        # group number of each subject
    date = '03-11'
    folder_name = './'

    for k in range(0, sub_num):
        for m in range(0, sub_group):
            # due to jaer's naming rule for aedat: the name includes generate date
            if((k+st_sub_idx) >= 20):
                date = '03-12'
            if((k+st_sub_idx) >= 42):
                date = '03-30'  
            if((k+st_sub_idx) >= 43):
                date = '03-31'

            data_file = ['./Davis346redColor-2019-' + date + 'Ts' + ("%02d" % i) +'-' + ("%02d" % j)
                +'.aedat'
                for i in range(st_sub_idx, sub_num+st_sub_idx+1)
                for j in range(1, sub_group+1)
                ]

            # img_file = 'C:/Users/UCRRR/Desktop/data/'
            # data_file = easygui.fileopenbox()
            csv_file =  ['./s' + ("%02d" % i) + '/raw_data_s' + ("%02d" % i) + '_' + ("%02d" % j) + '.csv'
                for i in range(st_sub_idx, sub_num+st_sub_idx+1)
                for j in range(1, sub_group+1)
                ] 

            s, x, y, pol = getDVSeventsDavis(data_file[k*sub_group+m])
            with open(csv_file[k*sub_group+m], "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "x", "y", "polarity"])
                # writer.writerows([s, x, y, pol])
                length = len(s)
                # length = 10
                for i in range(length):
                    lst =[s[i], x[i], y[i], int(pol[i])]
                    writer.writerow(lst)
                    lst.clear()

                    if i % 10000 == 0:
                        cnt = i/length *100
                        print("already finished %.1f %%" %cnt)
