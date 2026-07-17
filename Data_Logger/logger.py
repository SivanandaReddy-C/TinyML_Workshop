import serial
import csv

ser = serial.Serial('COM4',115200)

with open("dataset.csv","w",newline="") as f:

    writer = csv.writer(f)

    writer.writerow(
        ["Ax","Ay","Az","Gx","Gy","Gz"]
    )

    while True:

        line = ser.readline().decode().strip()

        values = line.split(",")

        if len(values)==6:

            writer.writerow(values)

            print(values)