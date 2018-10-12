//=======================================
// SOM���C���[����舵�����C���[�f�[�^
//=======================================
#ifndef __GRAVISBELL_I_LAYER_DATA_SOM_H__
#define __GRAVISBELL_I_LAYER_DATA_SOM_H__

#include"../Common/Common.h"
#include"../Common/ErrorCode.h"
#include"../Common/IODataStruct.h"
#include"../Common/Guiddef.h"
#include"../Common/ITemporaryMemoryManager.h"

#include"../SettingData/Standard/IData.h"

#include"./ILayerBase.h"
#include"./ILayerData.h"

namespace Gravisbell {
namespace Layer {

	class ILayerDataSOM : public ILayerData
	{
	public:
		/** �R���X�g���N�^ */
		ILayerDataSOM() : ILayerData(){}
		/** �f�X�g���N�^ */
		virtual ~ILayerDataSOM(){}


		//==================================
		// SOM�֘A����
		//==================================
	public:
		/** �}�b�v�T�C�Y���擾����.
			@return	�}�b�v�̃o�b�t�@����Ԃ�.(F32�z��̗v�f��) */
		virtual U32 GetMapSize()const = 0;

		/** �}�b�v�̃o�b�t�@���擾����.
			@param	o_lpMapBuffer	�}�b�v���i�[����z�X�g�������o�b�t�@. GetMapSize()�̖߂�l�̗v�f�����K�v. */
		virtual Gravisbell::ErrorCode GetMapBuffer(F32* o_lpMapBuffer)const = 0;
	};

}	// Layer
}	// Gravisbell

#endif