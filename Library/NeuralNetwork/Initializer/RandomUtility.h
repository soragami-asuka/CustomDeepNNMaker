//===================================
// ������Utility
//===================================
#ifndef __RANDOM_UTILITY_H__
#define __RANDOM_UTILITY_H__

#include<boost/random.hpp>

#include"Common/Common.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Random
	{
	private:
		boost::random::mt19937 gen;	/**< �����W�F�l���[�^ */

		boost::random::uniform_real_distribution<F32> distF;

	public:
		/** �R���X�g���N�^ */
		Random();
		/** �f�X�g���N�^ */
		virtual ~Random();

	public:
		/** ������.
			�����̎���Œ�l�Ŏw�肵�ē��o���������ꍇ�Ɏg�p����. */
		void Initialize(U32 seed);

	private:

		/** 0.0 �` 1.0�͈̔͂Œl���擾���� */
		F32 GetValue();

	public:
		/** ��l�������擾���� */
		F32 GetUniformValue(F32 min, F32 max);

		/** ���K�������擾����.
			@param	average	����
			@param	sigma	�W���΍����㕪�U */
		F32 GetNormalValue(F32 average, F32 sigma);

		/** �ؒf���K�������擾����.
			@param	average	����
			@param	sigma	�W���΍����㕪�U */
		F32 GetTruncatedNormalValue(F32 average, F32 sigma);
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif	__RANDOM_UTILITY_H__