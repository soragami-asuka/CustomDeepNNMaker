//===================================
// ������Utility
//===================================
#include"stdafx.h"

#include"RandomUtility.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#undef min
#undef max

/** �R���X�g���N�^ */
Random::Random()
		:	gen		((uint32_t)time(NULL))	/**< �����W�F�l���[�^ */

		,	distF	(0, 1)
{
}
/** �f�X�g���N�^ */
Random::~Random()
{
}


/** ������.
	�����̎���Œ�l�Ŏw�肵�ē��o���������ꍇ�Ɏg�p����. */
void Random::Initialize(U32 seed)
{
	this->gen = boost::random::mt19937(seed);
}

/** 0.0 �` 1.0�͈̔͂Œl���擾���� */
F32 Random::GetValue()
{
	return this->distF(this->gen);
}

/** ��l�������擾���� */
F32 Random::GetUniformValue(F32 min, F32 max)
{
	static boost::random::uniform_real_distribution<F32> dist(min, max);

	if(dist.min() != min || dist.max() != max)
		dist = boost::random::uniform_real_distribution<F32>(min, max);

	return dist(this->gen);
}

/** ���K�������擾���� */
F32 Random::GetNormalValue(F32 average, F32 sigma)
{
	static boost::random::normal_distribution<F32> dist(average, sigma);

	if(dist.mean() != average || dist.sigma() != sigma)
		dist = boost::random::normal_distribution<F32>(average, sigma);

	return dist(this->gen);
}

/** �ؒf���K�������擾����.
	@param	average	����
	@param	sigma	�W���΍����㕪�U */
F32 Random::GetTruncatedNormalValue(F32 average, F32 sigma)
{
	return std::max(-sigma, std::min(sigma, GetNormalValue(average, sigma)));
}