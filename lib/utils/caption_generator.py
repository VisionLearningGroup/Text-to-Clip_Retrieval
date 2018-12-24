import heapq
import math
import numpy as np

class Caption(object):
  """Represents a complete or partial caption."""

  def __init__(self, sentence, state, logprob, score, metadata=None):
    """Initializes the Caption.
    Args:
      sentence: List of word ids in the caption.
      state: Model state after generating the previous word.
      logprob: Log-probability of the caption.
      score: Score of the caption.
      metadata: Optional metadata associated with the partial sentence. If not
        None, a list of strings with the same length as 'sentence'.
    """
    self.sentence = sentence
    self.state = state
    self.logprob = logprob
    self.score = score
    self.metadata = metadata

  def __cmp__(self, other):
    """Compares Captions by score."""
    assert isinstance(other, Caption)
    if self.score == other.score:
      return 0
    elif self.score < other.score:
      return -1
    else:
      return 1
  
  # For Python 3 compatibility (__cmp__ is deprecated).
  def __lt__(self, other):
    assert isinstance(other, Caption)
    return self.score < other.score
  
  # Also for Python 3 compatibility.
  def __eq__(self, other):
    assert isinstance(other, Caption)
    return self.score == other.score


class TopN(object):
  """Maintains the top n elements of an incrementally provided set."""

  def __init__(self, n):
    self._n = n
    self._data = []

  def size(self):
    assert self._data is not None
    return len(self._data)

  def push(self, x):
    """Pushes a new element."""
    assert self._data is not None
    if len(self._data) < self._n:
      heapq.heappush(self._data, x)
    else:
      heapq.heappushpop(self._data, x)

  def extract(self, sort=False):
    """Extracts all elements from the TopN. This is a destructive operation.
    The only method that can be called immediately after extract() is reset().
    Args:
      sort: Whether to return the elements in descending sorted order.
    Returns:
      A list of data; the top n elements provided to the set.
    """
    assert self._data is not None
    data = self._data
    self._data = None
    if sort:
      data.sort(reverse=True)
    return data

  def reset(self):
    """Returns the TopN to an empty state."""
    self._data = []


class CaptionGenerator(object):
  """Class to generate captions from an image-to-text model."""

  def __init__(self,
               net,
               vocab,
               beam_size=3,
               max_caption_length=20,
               length_normalization_factor=0.0):
    """Initializes the generator.
    Args:
      model: Object encapsulating a trained image-to-text model. Must have
        methods feed_image() and inference_step(). For example, an instance of
        InferenceWrapperBase.
      vocab: A Vocabulary object.
      beam_size: Beam size to use when generating captions.
      max_caption_length: The maximum caption length before stopping the search.
      length_normalization_factor: If != 0, a number x such that captions are
        scored by logprob/length^x, rather than logprob. This changes the
        relative scores of captions depending on their lengths. For example, if
        x > 0 then longer captions will be favored.
    """
    self.vocab = vocab
    self.net = net

    self.beam_size = beam_size
    self.max_caption_length = max_caption_length
    self.length_normalization_factor = length_normalization_factor

  def beam_search(self, encoded_image):
    """Runs beam search caption generation on a single image.
    Args:
      sess: TensorFlow Session object.
      encoded_image: An encoded image string.
    Returns:
      A list of Caption sorted by descending score.
    """
    initial_beam = Caption(
        sentence=[0],
        state=[np.zeros((1,1,512)),
               np.zeros((1,1,512)),
               np.zeros((1,1,512)),
               np.zeros((1,1,512))],   # [lstm1_c, lstm1_h, lstm2_c, lstm2_h]
        logprob=0.0,
        score=0.0,
        metadata="")
    partial_captions = TopN(self.beam_size)
    partial_captions.push(initial_beam)
    complete_captions = TopN(self.beam_size)

    # Run beam search.
    for t in xrange(self.max_caption_length - 1):
      partial_captions_list = partial_captions.extract()
      partial_captions.reset()
      for i, partial_caption in enumerate(partial_captions_list):
        cont_sentence = np.array([1]) if len(partial_caption.sentence) > 1 else np.array([0])
        input_sentence = np.array([partial_caption.sentence[-1]])
        lstm1_c0 = partial_caption.state[0]
        lstm1_h0 = partial_caption.state[1]
        lstm2_c0 = partial_caption.state[2]
        lstm2_h0 = partial_caption.state[3]
        caption_fc6 = encoded_image

        self.net.forward(cont_sentence=cont_sentence,
                         input_sentence=input_sentence,
                         caption_fc6=caption_fc6,
                         lstm1_c0=lstm1_c0, lstm1_h0=lstm1_h0,
                         lstm2_c0=lstm2_c0, lstm2_h0=lstm2_h0)

        word_probs = self.net.blobs['probs'].data
        state = [self.net.blobs['lstm1_c1'].data, self.net.blobs['lstm1_h1'].data,
                 self.net.blobs['lstm2_c1'].data, self.net.blobs['lstm2_h1'].data]

        #print word_probs[0][0]
        #raw_input()
        words_and_probs = list(enumerate(word_probs[0][0]))
        words_and_probs.sort(key=lambda x: -x[1])
        #print words_and_probs
        words_and_probs = words_and_probs[0:self.beam_size]
        for w, p in words_and_probs:
          if p < 1e-12:
            continue
          sentence = partial_caption.sentence + [w]
          logprob = partial_caption.logprob + math.log(p)
          score = logprob
          metadata_list = partial_caption.metadata + self.vocab[w] + ' '
          #print sentence
          if w == 0:
            if self.length_normalization_factor > 0:
              score /= len(sentence)**self.length_normalization_factor
              #score /= len(sentence)**self.length_normalization_factor
              #score = np.max((score,16))
            beam = Caption(sentence, state, logprob, score, metadata_list[:-6])
            complete_captions.push(beam)
          else:
            beam = Caption(sentence, state, logprob, score, metadata_list)
            partial_captions.push(beam)
      if partial_captions.size() == 0:
        # We have run out of partial candidates; happens when beam_size = 1.
        break
             
    # If we have no complete captions then fall back to the partial captions.
    # But never output a mixture of complete and partial captions because a
    # partial caption could have a higher score than all the complete captions.
    if not complete_captions.size():
      complete_captions = partial_captions

    #print complete_captions.extract(sort=True)[0].metadata

    return complete_captions.extract(sort=True)[0].metadata
    
    # print "\n print all the sentenes in the complete_captions: \n............................"
    # all_sent = complete_captions.extract(sort=True)
    # for m in xrange(len(all_sent)):
    #   print all_sent[m].metadata 

    # return all_sent[m].metadata

