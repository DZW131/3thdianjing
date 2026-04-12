import xml.etree.cElementTree as et

tree=et.parse("/home/kd/Desktop/884.xml")
root=tree.getroot()
filename=root.find('filename').text
print(filename)
for Object in root.findall('object'):
    name=Object.find('name').text
    print(name)
    bndbox=Object.find('bndbox')
    xmin=bndbox.find('xmin').text
    ymin=bndbox.find('ymin').text
    xmax=bndbox.find('xmax').text
    ymax=bndbox.find('ymax').text
    print(xmin,ymin,xmax,ymax)
