#include <janice_io_opencv.h>

#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include <string>
#include <vector>

namespace
{

static inline JaniceError cv_mat_to_janice_image(cv::Mat& m, JaniceImage* _image)
{
    // Allocate a new image
    JaniceImage image = new JaniceImageType();

    // Set up the dimensions
    image->channels = m.channels();
    image->rows = m.rows;
    image->cols = m.cols;

    image->data = (uint8_t*) malloc(m.channels() * m.rows * m.cols);
    memcpy(image->data, m.data, m.channels() * m.rows * m.cols);
    image->owner = true;

    *_image = image;

    return JANICE_SUCCESS;
}

// ----------------------------------------------------------------------------
// JaniceMediaIterator

struct JaniceMediaIteratorStateTypeFM
{
    std::vector<JaniceImage> images;
    size_t pos;
};

JaniceError is_video(JaniceMediaIterator it, bool* video)
{
    *video = true; // Treat this as a video
    return JANICE_SUCCESS;
}

JaniceError get_frame_rate(JaniceMediaIterator, float*)
{
    return JANICE_INVALID_MEDIA;
}

JaniceError next(JaniceMediaIterator it, JaniceImage* image)
{
    JaniceMediaIteratorStateTypeFM* state = (JaniceMediaIteratorStateTypeFM*) it->_internal;

    if (state->pos == state->images.size())
        return JANICE_MEDIA_AT_END;

    try {
        *image = state->images[state->pos];
    } catch (...) {
        return JANICE_UNKNOWN_ERROR;
    }

    ++state->pos;

    return JANICE_SUCCESS;
}

JaniceError seek(JaniceMediaIterator it, uint32_t frame)
{
    JaniceMediaIteratorStateTypeFM* state = (JaniceMediaIteratorStateTypeFM*) it->_internal;

    if (frame >= state->images.size())
        return JANICE_BAD_ARGUMENT;

    state->pos = frame;

    return JANICE_SUCCESS;
}

JaniceError get(JaniceMediaIterator it, JaniceImage* image, uint32_t frame)
{
    JaniceMediaIteratorStateTypeFM* state = (JaniceMediaIteratorStateTypeFM*) it->_internal;

    if (frame >= state->images.size())
        return JANICE_BAD_ARGUMENT;

    try {
        *image = state->images[frame];
    } catch (...) {
        return JANICE_UNKNOWN_ERROR;
    }

    return JANICE_SUCCESS;
}

JaniceError tell(JaniceMediaIterator it, uint32_t* frame)
{
    JaniceMediaIteratorStateTypeFM* state = (JaniceMediaIteratorStateTypeFM*) it->_internal;

    if (state->pos == state->images.size())
        return JANICE_MEDIA_AT_END;

    *frame = state->pos;

    return JANICE_SUCCESS;
}

JaniceError free_image(JaniceImage* image)
{
    if (image && (*image)->owner)
        free((*image)->data);
    delete (*image);

    return JANICE_SUCCESS;
}

JaniceError free_iterator(JaniceMediaIterator* it)
{
    if (it && (*it)->_internal) {
        delete (JaniceMediaIteratorStateTypeFM*) (*it)->_internal;
        delete (*it);
        *it = nullptr;
    }

    return JANICE_SUCCESS;
}

JaniceError reset(JaniceMediaIterator it)
{
    JaniceMediaIteratorStateTypeFM* state = (JaniceMediaIteratorStateTypeFM*) it->_internal;
    state->pos = 0;

    return JANICE_SUCCESS;
}

} // anonymous namespace

// ----------------------------------------------------------------------------
// OpenCV I/O only, create a from cv::Mat opencv_io media iterator

JaniceError janice_io_opencv_create_frommat(cv::Mat im, JaniceMediaIterator *_it)
{
    JaniceMediaIterator it = new JaniceMediaIteratorType();

    it->is_video = &is_video;
    it->get_frame_rate =  &get_frame_rate;

    it->next = &next;
    it->seek = &seek;
    it->get  = &get;
    it->tell = &tell;

    it->free_image = &free_image;
    it->free       = &free_iterator;

    it->reset      = &reset;

    JaniceMediaIteratorStateTypeFM* state = new JaniceMediaIteratorStateTypeFM();
    JaniceImage jim;
    cv_mat_to_janice_image(im, &jim);
    state->images.push_back(jim);
    state->pos = 0;

    it->_internal = (void*) (state);

    *_it = it;

    return JANICE_SUCCESS;
}
