// WeightData.cpp : DLL �A�v���P�[�V�����p�ɃG�N�X�|�[�g�����֐����`���܂��B
//

#include "stdafx.h"

#include<map>

#include"Library/NeuralNetwork/WeightData.h"

#include"WeightData_Default_base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	struct WeightDataCreateFunc
	{
		IWeightData* (*func_cpu)(U32, U32);
		IWeightData* (*func_gpu)(U32, U32);
	};
	static const std::map<std::wstring, WeightDataCreateFunc> lpWeightDataCreateFunc =
	{
		{L"Default", {CreateWeightData_Default_CPU, CreateWeightData_Default_GPU}},
	};
	
	/** �������Ǘ��N���X */
	class WeightDataManager : public IWeightDataManager
	{
	public:
		/** �R���X�g���N�^ */
		WeightDataManager(){};
		/** �f�X�g���N�^ */
		virtual ~WeightDataManager(){}

	public:
		/** �d�݃f�[�^���쐬����(CPU) */
		IWeightData* CreateWeightData_CPU(const wchar_t i_weigthDataID[], U32 i_weightSize, U32 i_biasSize)
		{
			auto it_func = lpWeightDataCreateFunc.find(i_weigthDataID);
			if(it_func == lpWeightDataCreateFunc.end())
				return NULL;

			return it_func->second.func_cpu(i_weightSize, i_biasSize);
		}
		/** �d�݃f�[�^���쐬����(GPU) */
		IWeightData* CreateWeightData_GPU(const wchar_t i_weigthDataID[], U32 i_weightSize, U32 i_biasSize)
		{
			auto it_func = lpWeightDataCreateFunc.find(i_weigthDataID);
			if(it_func == lpWeightDataCreateFunc.end())
				return NULL;

			return it_func->second.func_gpu(i_weightSize, i_biasSize);
		}
	};

	/** �������Ǘ��N���X���擾���� */
	WeightData_API IWeightDataManager& GetWeightDataManager(void)
	{
		static WeightDataManager weightDataManager;

		return weightDataManager;
	}


}	// NeuralNetwork
}	// Layer
}	// Gravisbell
