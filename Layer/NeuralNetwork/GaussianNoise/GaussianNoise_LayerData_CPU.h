//======================================
// プーリングレイヤーのデータ
//======================================
#ifndef __GAUSSIANNOISE_LAYERDATA_CPU_H__
#define __GAUSSIANNOISE_LAYERDATA_CPU_H__

#include"GaussianNoise_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class GaussianNoise_LayerData_CPU : public GaussianNoise_LayerData_Base
	{
		friend class GaussianNoise_CPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		GaussianNoise_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~GaussianNoise_LayerData_CPU();


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif