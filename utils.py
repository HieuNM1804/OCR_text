import torch
import numpy as np
from collections import defaultdict


def ctc_decode(predictions, encoder, beam_width=5):
    if predictions.is_cuda:
        predictions = predictions.cpu()
    predictions = torch.nn.functional.log_softmax(predictions, dim=2).detach().numpy()
    decoded_batch = []
    for probs in predictions:
        beam = {(): (0.0, -float('inf'))}
        for t in range(len(probs)):
            next_beam = defaultdict(lambda: (-float('inf'), -float('inf')))
            top_k = min(beam_width, probs.shape[1])
            top_indices = np.argsort(probs[t])[-top_k:]
            for prefix, (p_b, p_nb) in beam.items():
                p_blank = probs[t][0]
                n_p_b, n_p_nb = next_beam[prefix]
                n_p_b = np.logaddexp(n_p_b, np.logaddexp(p_b, p_nb) + p_blank)
                next_beam[prefix] = (n_p_b, n_p_nb)
                for c in top_indices:
                    if c == 0: continue
                    p_char = probs[t][c]
                    if len(prefix) > 0 and prefix[-1] == c:
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_nb = np.logaddexp(n_p_nb, p_nb + p_char)
                        next_beam[prefix] = (n_p_b, n_p_nb)
                        new_prefix = prefix + (c,)
                        n_p_b, n_p_nb = next_beam[new_prefix]
                        n_p_nb = np.logaddexp(n_p_nb, p_b + p_char)
                        next_beam[new_prefix] = (n_p_b, n_p_nb)
                    else:
                        new_prefix = prefix + (c,)
                        n_p_b, n_p_nb = next_beam[new_prefix]
                        n_p_nb = np.logaddexp(n_p_nb, np.logaddexp(p_b, p_nb) + p_char)
                        next_beam[new_prefix] = (n_p_b, n_p_nb)
            sorted_beam = sorted(
                next_beam.items(),
                key=lambda x: np.logaddexp(x[1][0], x[1][1]),
                reverse=True
            )
            beam = dict(sorted_beam[:beam_width])
        best_prefix = max(beam.items(), key=lambda x: np.logaddexp(x[1][0], x[1][1]))[0]
        decoded_batch.append(encoder.decode(best_prefix))
    return decoded_batch
