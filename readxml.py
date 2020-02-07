# coding=utf-8
import os
import xml.dom.minidom


dict_in = 'E:/E - Course/course/2019-20-MSC-sem2/Dissertation/drive/1 Doors/Installed Doors/xml'
current_num = 0
total_num = len(os.listdir(dict_in))


for file in os.listdir(dict_in):
    read_xml = dict_in + '/' + file.strip()
    read_xml_fullname = os.path.basename(read_xml)
    read_xml_name, read_xml_format = os.path.splitext(read_xml_fullname)

    # open xml
    dom = xml.dom.minidom.parse(read_xml)
    root = dom.documentElement
    objs = root.getElementsByTagName('object')

    # iter every obj
    str_out = "lslm/" + str(read_xml_name) + ".jpg" + "         "
    for obj in objs:
        labels = obj.getElementsByTagName('name')

        # there are 4 labels
        if labels[0].childNodes[0].data == 'door':
            values = obj.getElementsByTagName('bndbox')
            x1 = values[0].getElementsByTagName("xmin")[0].childNodes[0].data
            y1 = values[0].getElementsByTagName("ymin")[0].childNodes[0].data
            x2 = values[0].getElementsByTagName("xmax")[0].childNodes[0].data
            y2 = values[0].getElementsByTagName("ymax")[0].childNodes[0].data
            str_out += '{"value":"%s","coordinate":[[%s,%s],[%s,%s]]}    ' % ('door', x1, y1, x2, y2)

        elif labels[0].childNodes[0].data == 'door_frame':
            values = obj.getElementsByTagName('bndbox')
            x1 = values[0].getElementsByTagName("xmin")[0].childNodes[0].data
            y1 = values[0].getElementsByTagName("ymin")[0].childNodes[0].data
            x2 = values[0].getElementsByTagName("xmax")[0].childNodes[0].data
            y2 = values[0].getElementsByTagName("ymax")[0].childNodes[0].data
            str_out += '{"value":"%s","coordinate":[[%s,%s],[%s,%s]]}    ' % ('door_frame', x1, y1, x2, y2)

        elif labels[0].childNodes[0].data == 'knob':
            values = obj.getElementsByTagName('bndbox')
            x1 = values[0].getElementsByTagName("xmin")[0].childNodes[0].data
            y1 = values[0].getElementsByTagName("ymin")[0].childNodes[0].data
            x2 = values[0].getElementsByTagName("xmax")[0].childNodes[0].data
            y2 = values[0].getElementsByTagName("ymax")[0].childNodes[0].data
            str_out += '{"value":"%s","coordinate":[[%s,%s],[%s,%s]]}    ' % ('knob', x1, y1, x2, y2)

        elif labels[0].childNodes[0].data == 'lock':
            values = obj.getElementsByTagName('bndbox')
            x1 = values[0].getElementsByTagName("xmin")[0].childNodes[0].data
            y1 = values[0].getElementsByTagName("ymin")[0].childNodes[0].data
            x2 = values[0].getElementsByTagName("xmax")[0].childNodes[0].data
            y2 = values[0].getElementsByTagName("ymax")[0].childNodes[0].data
            str_out += '{"value":"%s","coordinate":[[%s,%s],[%s,%s]]}    ' % ('lock', x1, y1, x2, y2)

    print(str_out)



