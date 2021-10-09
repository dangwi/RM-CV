#include "lightboard.hpp"

bool LightBoard::clear_flag;
LightBoard::result_code LightBoard::result;

float distance(RotatedRect &a, RotatedRect &b)
{
    return sqrt(pow(a.center.x - b.center.x, 2) + \
        pow(a.center.y - b.center.y, 2));
}

float distance(Point2f &a, Point2f &b)
{
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

float distance(const float &a, const float &b)
{
    return abs(a-b);
}



LightBoard::LightBoard(RotatedRect &l, RotatedRect &r, Point2f &lp):
    left(l), right(r), _score(0), last_point(lp)
{
    _center = (left.center + right.center) / 2;

    // 1 / ( ratio - 1)
    float ratio = max(left.size.area(), right.size.area()) / \
        min(left.size.area(), right.size.area());
    float ave_height = (left.size.height + right.size.height) / 2;
    _score += 1.3 / ( ratio - 0.8 );

    // distance buf
    if ( distance(left, right) < ave_height )
        _score -= 25. * ave_height / distance(left, right);

    float body_ratio1 = left.size.height / left.size.width,
        body_ratio2 = right.size.height / right.size.width;

    if ( abs(body_ratio1 - body_ratio2) > 2 )
        _score -= 20;

    // size buf
    if ( ratio < 1.6 ) {
        _score += (left.size.area() + right.size.area()) / 200;
    }

    // height buf
    _score += 100 / (abs(left.center.x - right.center.y) + 5);

    if ( last_point != Point2f(-1, -1) ) {
        _score += 200 / (distance(_center.y, last_point.y) + 3);
    }


}

LightBoard::~LightBoard()
{
}

double LightBoard::score() const 
{
    return _score;
}

void LightBoard::drawArmor(Mat &frame) {
    if ( last_point != Point2f(-1, -1) ) {
        if ( distance(_center.y, last_point.y) > 60 ) {
            set_clear();
            // _score -= distance(_center, last_point) * 0.1;
            result = NEW_TARGET;
            return;
        }
    }
    ellipse(frame, left, Scalar(0,255,255), FILLED);
    ellipse(frame, right, Scalar(0,255,255), FILLED);

    rectangle(frame, left.boundingRect2f().tl(), right.boundingRect2f().br(), \
        Scalar(0,0,255), 3);


    // circle(frame, _center, 10, Scalar(0,0,255), FILLED);

    // putText(frame, to_string(_score), _center, FONT_HERSHEY_COMPLEX, 1, Scalar(255,255,0), 1);
    result = SUCCESS;
}

Point2f LightBoard::center() const {
    return _center;
}

void LightBoard::set_clear(bool clr) {
    clear_flag = clr;
}

bool LightBoard::if_clear() {
    return clear_flag;
}