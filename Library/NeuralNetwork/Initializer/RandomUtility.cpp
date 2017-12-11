//===================================
// 乱数のUtility
//===================================
#include"stdafx.h"

#include"RandomUtility.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#undef min
#undef max

/** コンストラクタ */
Random::Random()
		:	gen		((uint32_t)time(NULL))	/**< 乱数ジェネレータ */

		,	distF	(0, 1)
{
}
/** デストラクタ */
Random::~Random()
{
}


/** 初期化.
	乱数の種を固定値で指定して頭出ししたい場合に使用する. */
void Random::Initialize(U32 seed)
{
	this->gen = boost::random::mt19937(seed);
}

/** 0.0 ～ 1.0の範囲で値を取得する */
F32 Random::GetValue()
{
	return this->distF(this->gen);
}

/** 一様乱数を取得する */
F32 Random::GetUniformValue(F32 min, F32 max)
{
	static boost::random::uniform_real_distribution<F32> dist(min, max);

	if(dist.min() != min || dist.max() != max)
		dist = boost::random::uniform_real_distribution<F32>(min, max);

	return dist(this->gen);
}

/** 正規乱数を取得する */
F32 Random::GetNormalValue(F32 average, F32 sigma)
{
	static boost::random::normal_distribution<F32> dist(average, sigma);

	if(dist.mean() != average || dist.sigma() != sigma)
		dist = boost::random::normal_distribution<F32>(average, sigma);

	return dist(this->gen);
}

/** 切断正規乱数を取得する.
	@param	average	平均
	@param	sigma	標準偏差＝√分散 */
F32 Random::GetTruncatedNormalValue(F32 average, F32 sigma)
{
	return std::max(-sigma, std::min(sigma, GetNormalValue(average, sigma)));
}