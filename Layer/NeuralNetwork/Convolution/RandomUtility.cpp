//===================================
// 乱数のUtility
//===================================
#include"stdafx.h"

#include"RandomUtility.h"

using namespace Utility;

#undef min
#undef max

/** コンストラクタ */
Random::Random()
		:	gen		((uint32_t)time(NULL))	/**< 乱数ジェネレータ */

		,	distI16	(0, 0xFFFE)
		,	distI8	(0, 0xFE)
		,	distI1	(0, 1)
		,	distF	(0, 1)
{
}
/** デストラクタ */
Random::~Random()
{
}

/** インスタンスの取得 */
Random& Random::GetInstance()
{
	static Random instance;

	return instance;
}

/** 初期化.
	乱数の種を固定値で指定して頭出ししたい場合に使用する. */
void Random::Initialize(unsigned __int32 seed)
{
	GetInstance().gen = boost::random::mt19937(seed);
}


/** 0～65535の範囲で値を取得する */
unsigned __int16 Random::GetValueShort()
{
	auto& random = GetInstance();

	return random.distI16(random.gen);
}

/** 0～255の範囲で値を取得する */
unsigned __int8 Random::GetValueBYTE()
{
	auto& random = GetInstance();

	return random.distI8(random.gen);
}

/** 0or1の範囲で値を取得する */
bool Random::GetValueBIT()
{
	auto& random = GetInstance();

	return random.distI1(random.gen) == 1;
}

/** 0.0 ～ 1.0の範囲で値を取得する */
double Random::GetValue()
{
	auto& random = GetInstance();

	return random.distF(random.gen);
}

/** min ～ maxの範囲で値を取得する */
double Random::GetValue(double min, double max)
{
	static boost::random::uniform_real_distribution<> dist(min, max);

	if(dist.min() != min || dist.max() != max)
		dist = boost::random::uniform_real_distribution<>(min, max);
	
	auto& random = GetInstance();

	return dist(random.gen);
}

/** 世紀乱数を取得する */
double Random::GetNormalValue(double average, double variance)
{
	double alpha = GetValue();
	double beta  = GetValue();

	double randomValue = sqrt(-2.0 * log(alpha)) * sin(2.0 * 3.1415 * beta);

	return randomValue * variance + average;
}