//===================================
// ������Utility
//===================================
#pragma once

#ifdef _CRTDBG_MAP_ALLOC
#undef new
#endif

#include<boost/random.hpp>

#ifdef _CRTDBG_MAP_ALLOC
#define new  ::new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif


namespace Utility
{
	class Random
	{
	private:
		boost::random::mt19937 gen;	/**< �����W�F�l���[�^ */

		boost::random::uniform_int_distribution<> distI16;
		boost::random::uniform_int_distribution<> distI8;
		boost::random::uniform_int_distribution<> distI1;
		boost::random::uniform_real_distribution<> distF;

	private:
		/** �R���X�g���N�^ */
		Random();
		/** �f�X�g���N�^ */
		virtual ~Random();

	private:
		/** �C���X�^���X�̎擾 */
		static Random& GetInstance();

	public:
		/** ������.
			�����̎���Œ�l�Ŏw�肵�ē��o���������ꍇ�Ɏg�p����. */
		static void Initialize(unsigned __int32 seed);

	public:
		/** 0�`65535�͈̔͂Œl���擾���� */
		static unsigned __int16 GetValueShort();

		/** 0�`255�͈̔͂Œl���擾���� */
		static unsigned __int8 GetValueBYTE();

		/** 0or1�͈̔͂Œl���擾���� */
		static bool GetValueBIT();

		/** 0.0 �` 1.0�͈̔͂Œl���擾���� */
		static double GetValue();

		/** min �` max�͈̔͂Œl���擾���� */
		static double GetValue(double min, double max);
	};
}
