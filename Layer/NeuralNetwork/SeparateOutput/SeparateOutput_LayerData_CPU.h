//======================================
// 出力信号分割レイヤーのデータ
//======================================
#ifndef __SEPARATEOUTPUT_LAYERDATA_CPU_H__
#define __SEPARATEOUTPUT_LAYERDATA_CPU_H__

#include"SeparateOutput_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class SeparateOutput_LayerData_CPU : public SeparateOutput_LayerData_Base
	{
		friend class SeparateOutput_CPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		SeparateOutput_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~SeparateOutput_LayerData_CPU();


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif