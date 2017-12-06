//===================================
// ������Utility
//===================================
#include"stdafx.h"

#include"RandomUtility.h"

using namespace Utility;

#undef min
#undef max

/** �R���X�g���N�^ */
Random::Random()
		:	gen		((uint32_t)time(NULL))	/**< �����W�F�l���[�^ */

		,	distI16	(0, 0xFFFE)
		,	distI8	(0, 0xFE)
		,	distI1	(0, 1)
		,	distF	(0, 1)
{
}
/** �f�X�g���N�^ */
Random::~Random()
{
}

/** �C���X�^���X�̎擾 */
Random& Random::GetInstance()
{
	static Random instance;

	return instance;
}

/** ������.
	�����̎���Œ�l�Ŏw�肵�ē��o���������ꍇ�Ɏg�p����. */
void Random::Initialize(unsigned __int32 seed)
{
	GetInstance().gen = boost::random::mt19937(seed);
}


/** 0�`65535�͈̔͂Œl���擾���� */
unsigned __int16 Random::GetValueShort()
{
	auto& random = GetInstance();

	return random.distI16(random.gen);
}

/** 0�`255�͈̔͂Œl���擾���� */
unsigned __int8 Random::GetValueBYTE()
{
	auto& random = GetInstance();

	return random.distI8(random.gen);
}

/** 0or1�͈̔͂Œl���擾���� */
bool Random::GetValueBIT()
{
	auto& random = GetInstance();

	return random.distI1(random.gen) == 1;
}

/** 0.0 �` 1.0�͈̔͂Œl���擾���� */
double Random::GetValue()
{
	auto& random = GetInstance();

	return random.distF(random.gen);
}

/** min �` max�͈̔͂Œl���擾���� */
double Random::GetValue(double min, double max)
{
	static boost::random::uniform_real_distribution<> dist(min, max);

	if(dist.min() != min || dist.max() != max)
		dist = boost::random::uniform_real_distribution<>(min, max);
	
	auto& random = GetInstance();

	return dist(random.gen);
}

/** ���I�������擾���� */
double Random::GetNormalValue(double average, double variance)
{
	double alpha = GetValue();
	double beta  = GetValue();

	double randomValue = sqrt(-2.0 * log(alpha)) * sin(2.0 * 3.1415 * beta);

	return randomValue * variance + average;
}