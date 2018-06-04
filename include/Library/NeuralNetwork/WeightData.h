//===============================================
// �œK�����[�`��
//===============================================
#ifndef __GRAVISBELL_LIBRARY_NN_WEIGHTDATA_H__
#define __GRAVISBELL_LIBRARY_NN_WEIGHTDATA_H__

#ifdef WEIGHTDATA_EXPORTS
#define WeightData_API __declspec(dllexport)
#else
#define WeightData_API __declspec(dllimport)
#ifndef GRAVISBELL_LIBRARY
#endif
#endif

#include"Layer/NeuralNetwork/IWeightData.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** �������Ǘ��N���X */
	class IWeightDataManager
	{
	public:
		/** �R���X�g���N�^ */
		IWeightDataManager(){};
		/** �f�X�g���N�^ */
		virtual ~IWeightDataManager(){}

	public:
		/** �d�݃f�[�^���쐬����(CPU) */
		virtual IWeightData* CreateWeightData_CPU(const wchar_t i_weigthDataID[], U32 i_weightSize, U32 i_biasSize) = 0;
		/** �d�݃f�[�^���쐬����(GPU) */
		virtual IWeightData* CreateWeightData_GPU(const wchar_t i_weigthDataID[], U32 i_weightSize, U32 i_biasSize) = 0;
	};

	/** �������Ǘ��N���X���擾���� */
	WeightData_API IWeightDataManager& GetWeightDataManager(void);

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
