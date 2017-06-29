//======================================
// 畳み込みニューラルネットワークのレイヤーデータ
//======================================
#ifndef __UpSampling_LAYERDATA_CPU_H__
#define __UpSampling_LAYERDATA_CPU_H__

#include"UpSampling_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class UpSampling_LayerData_CPU : public UpSampling_LayerData_Base
	{
		friend class UpSampling_CPU;

	private:
		// 本体
		std::vector<std::vector<NEURON_TYPE>>	lppNeuron;			/**< 各ニューロンの係数<ニューロン数, 入力数> */
		std::vector<NEURON_TYPE>				lpBias;				/**< ニューロンのバイアス<ニューロン数> */

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		UpSampling_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~UpSampling_LayerData_CPU();


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif