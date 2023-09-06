import math


def eps_decay(x: float):
  slope1 = 30_000
  start1 = 0.5
  stop1 = 0.01
  bottom_limiter = math.exp(-x / slope1)*(start1 - stop1) + stop1

  slope2 = 60_000
  start2 = 1
  stop2 = 0.25
  top_limiter = math.exp(-x / slope2)*(start2 - stop2) + stop2

  center = (top_limiter + bottom_limiter)/2

  slope3 = 300_000
  start3 = 2500
  stop3 = 8000
  frequency = math.exp(-x / slope3)*(start3 - stop3) + stop3

  shift = 4000
  amplitude = (top_limiter - bottom_limiter)/2
  wave = math.sin((x+shift)/frequency)*amplitude

  return center + wave
