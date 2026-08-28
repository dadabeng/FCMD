import math
import time

from scipy.ndimage import gaussian_filter


def print_current_loss(accelerator, start_time, niter_state, losses, epoch=None, inner_iter=None):
    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None:
        accelerator.print('epoch: %3d niter: %6d  inner_iter: %4d' % (epoch, niter_state, inner_iter), end=" ")

    now = time.time()
    message = '%s' % (as_minutes(now - start_time))

    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    accelerator.print(message)


def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)
