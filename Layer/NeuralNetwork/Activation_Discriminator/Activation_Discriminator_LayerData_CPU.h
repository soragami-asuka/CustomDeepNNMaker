//======================================
// 活性化関数のレイヤーデータ
//======================================
#ifndef __Activation_Discriminator_LAYERDATA_CPU_H__
#define __Activation_Discriminator_LAYERDATA_CPU_H__

#include"Activation_Discriminator_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Activation_Discriminator_LayerData_CPU : public Activation_Discriminator_LayerData_Base
	{
		friend class Activation_Discriminator_CPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		Activation_Discriminator_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~Activation_Discriminator_LayerData_CPU();


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