import cv2
import numpy as np
import os

def detectItem(canny_obj, height, width, iconSize):
    canny_v = list()
    index = 0
    # 水平裁切
    for ch in np.vsplit(canny_obj, height//iconSize):
        cv2.imwrite('{}.jpg'.format(index), ch)
        index += 1
        # 垂直裁切
        for i in np.hsplit(ch, width//iconSize):
            # cv2.imwrite('{}.png'.format(index), i)

            canny_v.append(i)
    canny_v = np.array(canny_v)

    print('len(canny_v) = {}'.format(len(canny_v)))
    icon_status = [False for i in range(len(canny_v))]  # 紀錄該區域內是否有東西 左至右上而下
    icon_status_index = 0
    for cv in canny_v:
        row_logic_or = np.full(iconSize, False)  # 宣告偵測物體寬度變數
        for a in cv:
            row_logic_or = np.logical_or(row_logic_or, a)
        # print(row_logic_or)

        cv_tran = cv.transpose()  # 翻轉 row and col

        col_logic_or = np.full(iconSize, False)  # 宣告偵測物體高度變數
        for a in cv_tran:
            col_logic_or = np.logical_or(col_logic_or, a)
        # print(col_logic_or)

        # 視為有東西的門檻 row 跟 col 連續4個點有東西時
        check_threshold = [True, True, True, True]
        if all([', '.join(map(str, check_threshold)) in ', '.join(map(str, row_logic_or)), ', '.join(map(str, check_threshold)) in ', '.join(map(str, col_logic_or))]):
            icon_status[icon_status_index] = True
        icon_status_index += 1

    return icon_status


def ObjectDetection(sourcePath: str, targetDir: str, resultDir: str):
    print('ObjectDetection()')
    Detect_num = 0
    Detect_itemSet = set()
    img_bgr = cv2.imread(sourcePath)
    (img_height, img_width, img_channels) = img_bgr.shape  # 讀取原圖size
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # 灰階

    # 逐張掃描Target圖片比對
    for targetFile in os.listdir(targetDir):
        if all([not targetFile.endswith('.jpg'), not targetFile.endswith('.png')]):
            print(
                'pass this file {}, because it didn\'t match .jpg or .png'.format(targetFile))
            continue
        target = os.path.join(targetDir, targetFile)
        # print('target = {}'.format(target))

        target = cv2.imread(target, 0)  # 以灰階read
        w, h = target.shape[::-1]
        res = cv2.matchTemplate(img_gray, target, cv2.TM_CCOEFF_NORMED)

        threshold = float(parameter['threshold'])
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            # 寫有detect到的object to log
            print('Detect {} => {}'.format(targetFile, pt))
            targetFile_temp = targetFile.replace(
                '.jpg', '').replace('.png', '').split('_')
            # print('targetFile_temp = {}'.format(targetFile_temp))
            Detect_itemSet.add(targetFile_temp[0])

            # 框圖
            cv2.rectangle(
                img_bgr, pt, (pt[0]+w, pt[1]+h), (0, 0, 255), 2)

            # 寫文字
            bottomLeftCornerOfText = (pt[0], pt[1])
            font, fontScale, fontColor, lineType = cv2.FONT_HERSHEY_PLAIN, 1, (
                0, 255, 255), 1
            cv2.putText(img_bgr, targetFile_temp[0],
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            Detect_num += 1
            # print('Detect_num = {}'.format(Detect_num))
            break  # 有比對成功即離開

    resulrImgPath = os.path.join(resultDir, os.path.basename(
        sourcePath).replace('.jpg', '_detect.jpg').replace('.png', '_detect.png'))
    print('resulrImgPath = {}'.format(resulrImgPath))
    cv2.imwrite(resulrImgPath, img_bgr)

    cv2.destroyAllWindows()
    return Detect_itemSet


if __name__ == "__main__":
    windowsDPI = 100

    parameter = {
        "detectfile": "C:/Users/MI/Desktop/CheckNotificationIcon/Notification_5.png",
        "examplepath": "Notification",
        "resultpath": "",
        "threshold": 0.8,
        "DPItoIconSize": {
            "100": 40,
            "125": 50,
            "150": 60
        },
        "Canny": {
            "threshold1": 10,
            "threshold2": 30
        }
    }

    img = cv2.imread(parameter['detectfile'])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(
        gray, parameter['Canny']['threshold1'], parameter['Canny']['threshold2'])
    height, width = canny.shape

    numberOfItem = 0
    # 此張截圖size符合對該DPI預期
    if all([height % parameter['DPItoIconSize'][str(windowsDPI)] == 0, width % parameter['DPItoIconSize'][str(windowsDPI)] == 0]):
        detectItemList = detectItem(
            canny, height, width, parameter['DPItoIconSize'][str(windowsDPI)])
        numberOfItem = detectItemList.count(True)
    else:
        cv2.destroyAllWindows()
        print('Expected size "{}" does not match'.format(
        parameter['DPItoIconSize'][str(windowsDPI)]))

    print('偵測到 {} 個 item'.format(numberOfItem))

    # Object detect
    try:
        detectitem_Set = ObjectDetection(
            parameter['detectfile'], parameter['examplepath'], parameter['resultpath'])
        detectitem_List = list(detectitem_Set)
        detectitem_List.sort()
        print('detectitem_List={}'.format(detectitem_List))
        print('Detect {} object!'.format(len(detectitem_List)))
    except Exception as err:
        print('Exception => {}'.format(err))
    else:
        print('ObjectDetection done!')

    cv2.imshow('img', img)

    result = np.hstack([gray, canny])
    cv2.imshow('gray canny', result)

    cv2.imwrite('canny.png', canny)
    cv2.imwrite('gray.png', gray)

    cv2.waitKey()
    cv2.destroyAllWindows()
