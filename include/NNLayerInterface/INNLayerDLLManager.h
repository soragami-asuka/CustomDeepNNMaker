//=======================================
// ���C���[DLL�N���X
//=======================================
#ifndef __GRAVISBELL_I_NN_LAYER_DLL_MANAGER_H__
#define __GRAVISBELL_I_NN_LAYER_DLL_MANAGER_H__

#include"Common/Guiddef.h"
#include"Common/ErrorCode.h"

#include"INNLayerDLL.h"

namespace Gravisbell {
namespace NeuralNetwork {

	class ILayerDLLManager
	{
	public:
		/** �R���X�g���N�^ */
		ILayerDLLManager(){}
		/** �f�X�g���N�^ */
		virtual ~ILayerDLLManager(){}

	public:
		/** DLL��ǂݍ���ŁA�Ǘ��ɒǉ�����.
			@param szFilePath		�ǂݍ��ރt�@�C���̃p�X.
			@param o_addLayerCode	�ǉ����ꂽGUID�̊i�[��A�h���X.
			@return	���������ꍇ0���Ԃ�. */
		virtual ErrorCode ReadLayerDLL(const wchar_t szFilePath[], GUID& o_addLayerCode) = 0;
		/** DLL��ǂݍ���ŁA�Ǘ��ɒǉ�����.
			@param szFilePath	�ǂݍ��ރt�@�C���̃p�X.
			@return	���������ꍇ0���Ԃ�. */
		virtual ErrorCode ReadLayerDLL(const wchar_t szFilePath[]) = 0;

		/** �Ǘ����Ă��郌�C���[DLL�̐����擾���� */
		virtual unsigned int GetLayerDLLCount()const = 0;
		/** �Ǘ����Ă��郌�C���[DLL��ԍ��w��Ŏ擾����.
			@param	num	�擾����DLL�̊Ǘ��ԍ�.
			@return ���������ꍇ��DLL�N���X�̃A�h���X. ���s�����ꍇ��NULL */
		virtual const ILayerDLL* GetLayerDLLByNum(unsigned int num)const = 0;
		/** �Ǘ����Ă��郌�C���[DLL��guid�w��Ŏ擾����.
			@param guid	�擾����DLL��GUID.
			@return ���������ꍇ��DLL�N���X�̃A�h���X. ���s�����ꍇ��NULL */
		virtual const ILayerDLL* GetLayerDLLByGUID(GUID i_layerCode)const = 0;

		/** ���C���[DLL���폜����. */
		virtual ErrorCode EraseLayerDLL(GUID i_layerCode) = 0;
	};

}	// NeuralNetwork
}	// Gravisbell

#endif