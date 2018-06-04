// WeightData.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
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
	
	/** 初期化管理クラス */
	class WeightDataManager : public IWeightDataManager
	{
	public:
		/** コンストラクタ */
		WeightDataManager(){};
		/** デストラクタ */
		virtual ~WeightDataManager(){}

	public:
		/** 重みデータを作成する(CPU) */
		IWeightData* CreateWeightData_CPU(const wchar_t i_weigthDataID[], U32 i_weightSize, U32 i_biasSize)
		{
			auto it_func = lpWeightDataCreateFunc.find(i_weigthDataID);
			if(it_func == lpWeightDataCreateFunc.end())
				return NULL;

			return it_func->second.func_cpu(i_weightSize, i_biasSize);
		}
		/** 重みデータを作成する(GPU) */
		IWeightData* CreateWeightData_GPU(const wchar_t i_weigthDataID[], U32 i_weightSize, U32 i_biasSize)
		{
			auto it_func = lpWeightDataCreateFunc.find(i_weigthDataID);
			if(it_func == lpWeightDataCreateFunc.end())
				return NULL;

			return it_func->second.func_gpu(i_weightSize, i_biasSize);
		}
	};

	/** 初期化管理クラスを取得する */
	WeightData_API IWeightDataManager& GetWeightDataManager(void)
	{
		static WeightDataManager weightDataManager;

		return weightDataManager;
	}


}	// NeuralNetwork
}	// Layer
}	// Gravisbell
